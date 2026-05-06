"""Assemble the final per-language picked predictions into one CodaBench submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from utils import counting, mfr, zip_files_flat


def load_test_predictions(model: str, lang: str) -> list[dict] | None:
    p = Path("outputs") / model / lang / "predictions_test.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fallback_mfr(lang: str, train_split, test_items):
    items = [dict(x) for x in train_split if x["lang"] == lang]
    counts = counting(items)
    return [{"raw": t["raw"], "pred": mfr(t["raw"], counts), "lang": lang} for t in test_items]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--picker", default="outputs/picker.json")
    p.add_argument("--out-dir", default="outputs/final")
    args = p.parse_args()

    decisions = json.load(open(args.picker, encoding="utf-8"))["decisions"]
    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")
    test_by_lang: dict[str, list[dict]] = {}
    for x in ds["test"]:
        test_by_lang.setdefault(x["lang"], []).append(dict(x))

    final = []
    for lang in sorted(test_by_lang.keys()):
        pick = decisions.get(lang, "mfr")
        if pick == "mfr":
            recs = fallback_mfr(lang, ds["train"], test_by_lang[lang])
        else:
            recs = load_test_predictions(pick, lang)
            if recs is None:
                print(f"[submit] WARN: no test preds for {pick}/{lang}; falling back to MFR")
                recs = fallback_mfr(lang, ds["train"], test_by_lang[lang])
        print(f"{lang:<6} -> {pick:<5}  ({len(recs)} sentences)")
        final.extend(recs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)
    print(f"\nWrote {len(final)} records -> {out_json}")

    zip_path = str(out_dir / "submission.zip")
    zip_files_flat(str(out_dir), zip_path, flag="-j")


if __name__ == "__main__":
    main()
