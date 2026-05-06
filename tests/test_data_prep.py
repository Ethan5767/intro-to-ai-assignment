import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep import to_token_to_token, CONTEXT


def test_basic_pair_emits_one_example_per_word():
    item = {"raw": ["i", "luv", "u"], "norm": ["I", "love", "you"], "lang": "en"}
    examples = to_token_to_token([item], context=2)
    assert len(examples) == 3
    assert examples[1]["input"] == "<L> i </L> <T> luv </T> <R> u </R>"
    assert examples[1]["output"] == "love"
    assert examples[1]["lang"] == "en"


def test_context_window_truncates_correctly():
    item = {"raw": ["a", "b", "c", "d", "e"], "norm": ["a", "b", "c", "d", "e"], "lang": "en"}
    examples = to_token_to_token([item], context=2)
    assert examples[2]["input"] == "<L> a b </L> <T> c </T> <R> d e </R>"


def test_edge_words_have_empty_context_blocks():
    item = {"raw": ["x", "y"], "norm": ["x", "y"], "lang": "en"}
    examples = to_token_to_token([item], context=3)
    assert examples[0]["input"] == "<L> </L> <T> x </T> <R> y </R>"
    assert examples[1]["input"] == "<L> x </L> <T> y </T> <R> </R>"


def test_multi_word_gold_preserved_as_single_string():
    item = {"raw": ["gonna"], "norm": ["going to"], "lang": "en"}
    examples = to_token_to_token([item], context=3)
    assert examples[0]["output"] == "going to"


def test_default_context_constant_is_three():
    assert CONTEXT == 3
