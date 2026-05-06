import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picker import pick_best


def test_picks_highest_err_when_gap_is_meaningful():
    scores = {
        "en": {"mfr": 0.50, "byt5": 0.61, "mt5": 0.55},
        "ko": {"mfr": 0.06, "byt5": -0.02, "mt5": 0.04},
    }
    assert pick_best(scores) == {"en": "byt5", "ko": "mfr"}


def test_missing_models_skipped():
    scores = {"th": {"mfr": 0.42, "byt5": 0.07}}  # mt5 absent
    assert pick_best(scores) == {"th": "mfr"}


def test_ties_break_with_priority_order():
    scores = {"x": {"mfr": 0.50, "byt5": 0.50, "mt5": 0.50}}
    assert pick_best(scores) == {"x": "byt5"}


def test_threshold_overrides_close_calls():
    # gap < 2 ERR points => prefer ByT5 (the workhorse default)
    close = {"y": {"mfr": 0.401, "byt5": 0.395, "mt5": 0.30}}
    assert pick_best(close, min_gap=0.02) == {"y": "byt5"}
    # gap >= 2 ERR points => switch to MFR
    big = {"z": {"mfr": 0.42, "byt5": 0.39, "mt5": 0.30}}
    assert pick_best(big, min_gap=0.02) == {"z": "mfr"}
