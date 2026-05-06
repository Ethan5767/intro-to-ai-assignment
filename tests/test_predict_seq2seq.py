import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict_seq2seq import reassemble_predictions


def test_reassemble_groups_by_sent_idx():
    examples = [
        {"sent_idx": 0, "word_idx": 0, "lang": "en"},
        {"sent_idx": 0, "word_idx": 1, "lang": "en"},
        {"sent_idx": 1, "word_idx": 0, "lang": "en"},
    ]
    decoded = ["I", "love", "Hi"]
    raw_sents = [["i", "luv"], ["hi"]]
    out = reassemble_predictions(examples, decoded, raw_sents, lang="en")
    assert out == [
        {"raw": ["i", "luv"], "pred": ["I", "love"], "lang": "en"},
        {"raw": ["hi"], "pred": ["Hi"], "lang": "en"},
    ]


def test_reassemble_preserves_split_outputs_as_one_string():
    examples = [{"sent_idx": 0, "word_idx": 0, "lang": "en"}]
    out = reassemble_predictions(examples, ["going to"], [["gonna"]], lang="en")
    assert out[0]["pred"] == ["going to"]


def test_reassemble_orders_by_word_idx_when_unsorted():
    examples = [
        {"sent_idx": 0, "word_idx": 1, "lang": "en"},
        {"sent_idx": 0, "word_idx": 0, "lang": "en"},
    ]
    out = reassemble_predictions(examples, ["B", "A"], [["a", "b"]], lang="en")
    assert out[0]["pred"] == ["A", "B"]
