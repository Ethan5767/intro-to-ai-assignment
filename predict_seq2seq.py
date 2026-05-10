"""Run inference for a fine-tuned ByT5/mT5 checkpoint and emit predictions.json."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_prep import to_token_to_token, annotate_sent_idx


def reassemble_predictions(examples, decoded, raw_sents, lang: str):
    """Group per-word predictions back into sentences."""
    by_sent: dict[int, dict[int, str]] = defaultdict(dict)
    for ex, pred in zip(examples, decoded):
        by_sent[ex["sent_idx"]][ex["word_idx"]] = pred
    out = []
    for sidx, raw in enumerate(raw_sents):
        words_by_idx = by_sent.get(sidx, {})
        pred = [words_by_idx.get(i, raw[i]) for i in range(len(raw))]
        out.append({"raw": raw, "pred": pred, "lang": lang})
    return out


def generate(model, tokenizer, examples, batch_size: int, max_in: int, max_out: int, num_beams: int):
    model.eval()
    device = next(model.parameters()).device
    decoded = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        enc = tokenizer([e["input"] for e in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_in).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_out, num_beams=num_beams)
        for ids in out:
            decoded.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return decoded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lang", required=True)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--out-json", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-in", type=int, default=256)
    p.add_argument("--max-out", type=int, default=64)
    p.add_argument("--num-beams", type=int, default=4)
    args = p.parse_args()

    print(f"[predict] {args.ckpt} on {args.lang}/{args.split}")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt).to("cuda")

    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")

    # 해당 언어의 모든 아이템을 일단 가져옵니다.
    all_items = [dict(x) for x in ds["train"] if x["lang"] == args.lang]

    if args.split == "validation" and not any(x["lang"] == args.lang for x in ds["validation"]):
        # 공식 validation이 없는 경우 (trde 등), train의 마지막 10%를 가져옴
        n = len(all_items)
        cut = int(n * 0.9)
        items = all_items[cut:]
        print(f"[predict] No official validation set found. Using held-out 10% from train ({len(items)} items).")
    else:
        # 공식 split이 존재하면 기존 방식대로 진행
        items = [dict(x) for x in ds[args.split] if x["lang"] == args.lang]
    
    raw_sents = [it["raw"] for it in items]
    if not items:
        print(f"[predict] WARNING: no items for {args.lang}/{args.split}")
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    annotated = annotate_sent_idx(items)
    examples = to_token_to_token(annotated)
    print(f"[predict] {len(items)} sentences, {len(examples)} word-level examples")

    decoded = generate(model, tokenizer, examples, args.batch_size, args.max_in, args.max_out, args.num_beams)
    records = reassemble_predictions(examples, decoded, raw_sents, lang=args.lang)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"[predict] wrote {len(records)} sentence records -> {args.out_json}")


if __name__ == "__main__":
    main()
