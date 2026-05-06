"""Drive train + predict across languages for one model variant.

Idempotent: if the `<ckpt-root>/<model>/<lang>/best/` directory already exists,
training is skipped. Per-language predictions land in
`outputs/{byt5,mt5}/<lang>/predictions_{dev,test}.json` and per-model dev ERR
is summarised in `outputs/summary_<model>.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

from utils import evaluate

ALL_LANGS = ["da", "de", "en", "es", "hr", "id", "iden", "it", "ja", "ko",
             "nl", "sl", "sr", "th", "tr", "trde", "vi"]


def score(pred_path: str, lang: str, split: str = "validation"):
    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")
    items = [dict(x) for x in ds[split] if x["lang"] == lang]
    if not items:
        return None
    raw = [it["raw"] for it in items]
    gold = [it["norm"] for it in items]
    with open(pred_path, encoding="utf-8") as f:
        preds = [r["pred"] for r in json.load(f)]
    lai, acc, err = evaluate(raw, gold, preds, info=False)
    return {"lai": lai, "acc": acc, "err": err}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["byt5-small", "mt5-small"])
    p.add_argument("--langs", nargs="*", default=ALL_LANGS)
    p.add_argument("--ckpt-root", default=os.environ.get("MLN_CKPT_ROOT", "./ckpts"))
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit-train", type=int, default=-1,
                   help="cap training examples (use for fast Colab smoke runs)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="retrain even if a checkpoint already exists")
    args = p.parse_args()

    summary = []
    for lang in args.langs:
        ckpt = Path(args.ckpt_root) / args.model / lang / "best"
        model_short = args.model.split("-")[0]  # byt5-small -> byt5
        pred_dev = Path(args.out_root) / model_short / lang / "predictions_dev.json"
        pred_test = Path(args.out_root) / model_short / lang / "predictions_test.json"

        if ckpt.exists() and not args.no_skip_existing:
            print(f"\n=== SKIP TRAIN {args.model}/{lang} (ckpt exists) ===")
        else:
            print(f"\n=== TRAIN {args.model} on {lang} ===")
            cmd = [sys.executable, "train_seq2seq.py",
                   "--model", args.model, "--lang", lang,
                   "--epochs", str(args.epochs),
                   "--ckpt-root", args.ckpt_root]
            if args.limit_train > 0:
                cmd += ["--limit-train", str(args.limit_train)]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[driver] train failed for {lang}; skipping rest of this lang")
                continue

        for split, out in (("validation", pred_dev), ("test", pred_test)):
            if out.exists() and not args.no_skip_existing:
                print(f"[driver] {out} exists; skipping predict")
                continue
            r = subprocess.run([sys.executable, "predict_seq2seq.py",
                                "--ckpt", str(ckpt), "--lang", lang,
                                "--split", split, "--out-json", str(out)])
            if r.returncode != 0:
                print(f"[driver] predict failed for {lang}/{split}")

        s = score(str(pred_dev), lang) if pred_dev.exists() else None
        if s:
            print(f"[score] {lang}: ERR={s['err']*100:.2f}")
            summary.append({"model": args.model, "lang": lang, **s})

    out_csv = Path(args.out_root) / f"summary_{args.model}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
        print(f"\n[driver] summary -> {out_csv}")
    else:
        print("\n[driver] no scoreable runs (no validation split for the chosen langs)")


if __name__ == "__main__":
    main()
