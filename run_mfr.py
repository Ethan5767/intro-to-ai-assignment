"""MFR baseline end-to-end: train per-language counts, score on dev, predict on test, zip."""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from utils import counting, mfr, evaluate, zip_files_flat

DATASET = "weerayut/multilexnorm2026-dev-pub"
OUT_DIR = Path(__file__).parent / "outputs" / "mfr_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def group_by_lang(split):
    by_lang = defaultdict(list)
    for ex in split:
        by_lang[ex["lang"]].append(ex)
    return by_lang


def main():
    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET)
    train_by_lang = group_by_lang(ds["train"])
    val_by_lang = group_by_lang(ds["validation"])
    test_by_lang = group_by_lang(ds["test"])

    langs = sorted(train_by_lang.keys())
    print(f"Languages in train: {langs}\n")

    # Per-language MFR counts
    counts_by_lang = {l: counting(train_by_lang[l]) for l in langs}

    # Score on validation where available
    print(f"{'lang':<6}{'n_val':>8}{'LAI':>8}{'Acc':>8}{'ERR':>8}")
    print("-" * 40)
    err_by_lang = {}
    val_macro = []
    for l in langs:
        if l not in val_by_lang:
            print(f"{l:<6}{'(no val)':>8}")
            continue
        items = val_by_lang[l]
        raw = [it["raw"] for it in items]
        gold = [it["norm"] for it in items]
        pred = [mfr(r, counts_by_lang[l]) for r in raw]
        lai, acc, err = evaluate(raw, gold, pred, info=False)
        err_by_lang[l] = err
        val_macro.append(err)
        print(f"{l:<6}{len(items):>8}{lai*100:>8.2f}{acc*100:>8.2f}{err*100:>8.2f}")
    print("-" * 40)
    if val_macro:
        print(f"Macro-avg ERR over {len(val_macro)} val langs: {sum(val_macro)/len(val_macro)*100:.2f}\n")

    # Per-language MFR summary (consumed by picker.py)
    summary_csv = OUT_DIR.parent / "summary_mfr.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "lang", "lai", "acc", "err"])
        for l, e in err_by_lang.items():
            items = val_by_lang[l]
            raw = [it["raw"] for it in items]
            gold = [it["norm"] for it in items]
            pred = [mfr(r, counts_by_lang[l]) for r in raw]
            lai, acc, _ = evaluate(raw, gold, pred, info=False)
            w.writerow(["mfr", l, lai, acc, e])
    print(f"Wrote {summary_csv}")

    # Predict on test (all 17 langs)
    print("Predicting on test ...")
    records = []
    for l in sorted(test_by_lang.keys()):
        items = test_by_lang[l]
        for it in items:
            pred = mfr(it["raw"], counts_by_lang.get(l, {}))
            records.append({"raw": it["raw"], "pred": pred, "lang": l})

    out_json = OUT_DIR / "predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Wrote {len(records)} test records -> {out_json}")

    # Zip flat for CodaBench
    zip_path = str(OUT_DIR.parent / "submission_mfr_dev.zip")
    zip_files_flat(str(OUT_DIR), zip_path, flag="-j")


if __name__ == "__main__":
    main()
