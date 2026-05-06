"""Per-language model picker: chooses the dev-best ERR model among MFR/ByT5/mT5.

Default behaviour: pick ByT5 (the established workhorse) unless another model
beats it by `min_gap` absolute ERR. The threshold prevents picker overfitting on
small noisy dev sets.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRIORITY = ("byt5", "mt5", "mfr")


def pick_best(scores: dict[str, dict[str, float]], min_gap: float = 0.02) -> dict[str, str]:
    out: dict[str, str] = {}
    for lang, by_model in scores.items():
        default = next((m for m in PRIORITY if m in by_model), None)
        if default is None:
            continue
        default_err = by_model[default]
        winner = default
        winner_err = default_err
        for m, v in by_model.items():
            if m == default:
                continue
            if v - default_err >= min_gap and v > winner_err:
                winner = m
                winner_err = v
        tied = [m for m, v in by_model.items() if abs(v - winner_err) < 1e-9]
        if len(tied) > 1:
            winner = next(m for m in PRIORITY if m in tied)
        out[lang] = winner
    return out


def load_summary(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["lang"]] = float(row["err"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--mfr-summary", default="outputs/summary_mfr.csv")
    p.add_argument("--out-json", default="outputs/picker.json")
    p.add_argument("--min-gap", type=float, default=0.02)
    args = p.parse_args()

    out_root = Path(args.out_root)
    byt5 = load_summary(out_root / "summary_byt5-small.csv")
    mt5 = load_summary(out_root / "summary_mt5-small.csv")
    mfr = load_summary(Path(args.mfr_summary))

    langs = sorted(set(byt5) | set(mt5) | set(mfr))
    scores: dict[str, dict[str, float]] = {}
    for l in langs:
        s = {}
        if l in mfr:
            s["mfr"] = mfr[l]
        if l in byt5:
            s["byt5"] = byt5[l]
        if l in mt5:
            s["mt5"] = mt5[l]
        if s:
            scores[l] = s

    decisions = pick_best(scores, min_gap=args.min_gap)
    print(f"{'lang':<6}{'mfr':>8}{'byt5':>8}{'mt5':>8}  picked")
    for l in sorted(decisions):
        s = scores[l]
        print(f"{l:<6}{s.get('mfr', float('nan'))*100:>8.2f}"
              f"{s.get('byt5', float('nan'))*100:>8.2f}"
              f"{s.get('mt5', float('nan'))*100:>8.2f}  {decisions[l]}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"decisions": decisions, "scores": scores}, f, indent=2, ensure_ascii=False)
    print(f"[picker] -> {args.out_json}")


if __name__ == "__main__":
    main()
