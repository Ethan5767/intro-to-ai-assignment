"""Fine-tune ByT5 or mT5 per language in UFAL token-to-token format."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from data_prep import to_token_to_token, annotate_sent_idx, SPECIALS


MODEL_PRESETS = {
    "byt5-small": "google/byt5-small",
    "mt5-small": "google/mt5-small",
}


def build_examples(hf_split, lang: str):
    items = annotate_sent_idx([dict(x) for x in hf_split if x["lang"] == lang])
    return to_token_to_token(items)


def tokenize_dataset(examples, tokenizer, max_in: int, max_out: int):
    def _tok(batch):
        inputs = tokenizer(batch["input"], max_length=max_in, truncation=True)
        labels = tokenizer(text_target=batch["output"], max_length=max_out, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    ds = Dataset.from_list(examples)
    return ds.map(_tok, batched=True, remove_columns=ds.column_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_PRESETS.keys()))
    p.add_argument("--lang", required=True)
    p.add_argument("--ckpt-root", default=os.environ.get("MLN_CKPT_ROOT", "./ckpts"),
                   help="where checkpoints live; override with $MLN_CKPT_ROOT")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit-eval", type=int, default=1000, help="cap val examples for per-epoch eval")
    p.add_argument("--max-steps", type=int, default=-1, help="for smoke testing")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-in", type=int, default=256)
    p.add_argument("--max-out", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=-1, help="cap train examples")
    p.add_argument("--patience", type=int, default=2)
    args = p.parse_args()

    model_id = MODEL_PRESETS[args.model]
    out_dir = Path(args.ckpt_root) / args.model / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] model={model_id} lang={args.lang} -> {out_dir}")

    print("[train] loading dataset")
    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_ex = build_examples(ds["train"], args.lang)
    if args.limit_train > 0:
        train_ex = train_ex[: args.limit_train]
        print(f"[train] limited train examples to {len(train_ex)}")
    val_ex = build_examples(ds["validation"], args.lang) if any(x["lang"] == args.lang for x in ds["validation"]) else None
    if val_ex is None:
        n = len(train_ex)
        cut = int(n * 0.9)
        val_ex = train_ex[cut:]
        train_ex = train_ex[:cut]
        print(f"[train] no dev split for {args.lang}; held out {len(val_ex)} from train")
    if args.limit_eval > 0 and len(val_ex) > args.limit_eval:
        val_ex = val_ex[: args.limit_eval]
        print(f"[train] capped eval examples to {len(val_ex)}")
    print(f"[train] train_examples={len(train_ex)} val_examples={len(val_ex)}")

    print("[train] loading tokenizer + model (no embedding resize; SPECIALS go through as bytes)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    train_ds = tokenize_dataset(train_ex, tokenizer, args.max_in, args.max_out)
    val_ds = tokenize_dataset(val_ex, tokenizer, args.max_in, args.max_out)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        optim="adafactor",
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))

    metrics_path = out_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"[train] saved -> {out_dir/'best'} (metrics: {metrics_path})")


if __name__ == "__main__":
    main()
