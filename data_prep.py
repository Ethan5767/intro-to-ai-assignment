"""Convert MultiLexNorm sentences to UFAL token-to-token training examples."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

CONTEXT = 3
SPECIALS = ("<L>", "</L>", "<T>", "</T>", "<R>", "</R>")


def _format_example(left: list[str], target: str, right: list[str]) -> str:
    parts = ["<L>"] + left + ["</L>", "<T>", target, "</T>", "<R>"] + right + ["</R>"]
    return " ".join(parts)


def to_token_to_token(items: Iterable[dict], context: int = CONTEXT) -> list[dict]:
    """Flatten sentences into per-word (input, output) pairs."""
    out = []
    for item in items:
        raw = item["raw"]
        gold = item["norm"]
        lang = item["lang"]
        n = len(raw)
        for i in range(n):
            left = raw[max(0, i - context):i]
            right = raw[i + 1:i + 1 + context]
            out.append({
                "input": _format_example(left, raw[i], right),
                "output": gold[i],
                "lang": lang,
                "sent_idx": item.get("sent_idx", -1),
                "word_idx": i,
            })
    return out


def split_by_lang(items: Iterable[dict]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for it in items:
        by[it["lang"]].append(it)
    return dict(by)


def annotate_sent_idx(items: Iterable[dict]) -> list[dict]:
    """Attach a per-language sentence index for later alignment."""
    counters: dict[str, int] = defaultdict(int)
    out = []
    for it in items:
        idx = counters[it["lang"]]
        counters[it["lang"]] += 1
        out.append({**it, "sent_idx": idx})
    return out
