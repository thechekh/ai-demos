"""The from-scratch BPE round-trips any text, and its merges behave as the article says."""

import random

import pytest

from tokens import LETTERS, PRE_SPLIT, encode_bpe, merge, train_bpe

TEXT = ("the cat sat on the mat. the cat ate the rat. then the cat sat on the hat, "
        "and the rat ran. " * 20)


def test_merge_joins_every_occurrence_of_a_pair():
    word = (b"a", b"b", b"a", b"b", b"c")
    assert merge(word, (b"a", b"b")) == (b"ab", b"ab", b"c")
    assert merge(word, (b"b", b"c")) == (b"a", b"b", b"a", b"bc")


def test_training_merges_the_most_frequent_pair_first():
    learned = train_bpe(TEXT, 5)
    counts = {}
    for w in PRE_SPLIT.findall(TEXT):
        b = w.encode()
        for pair in zip(b, b[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    top = max(counts.values())
    first = learned[0]
    assert counts[(first[0][0], first[1][0])] == top


@pytest.mark.parametrize("seed", range(5))
def test_encoding_loses_nothing(seed):
    rng = random.Random(seed)
    learned = train_bpe(TEXT, 40)
    alphabet = "abc xyz,.!?éЖ中\U0001f44d\n"
    text = "".join(rng.choice(alphabet) for _ in range(300))
    assert b"".join(encode_bpe(text, learned)) == text.encode("utf-8")


def test_more_merges_mean_fewer_tokens_on_the_training_text():
    lengths = [len(encode_bpe(TEXT, train_bpe(TEXT, m))) for m in (0, 10, 40)]
    assert lengths[0] == len(TEXT.encode()) and lengths[0] > lengths[1] > lengths[2]


def test_the_letter_words_are_balanced():
    counts = sorted(w.count(c) for w, c in LETTERS)
    assert counts == [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
