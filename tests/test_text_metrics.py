"""Each hand-written metric agrees with its reference library."""

import random

import pytest
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from text_metrics import (BertScorer, bleu, exact_match, lcs, ranks, rouge_l, rouge_n, spearman, token_f1,
                          words)

SENTENCES = ["The drug is safe for children.", "The medicine is safe for kids.", "Children for safe is drug the.",
             "Stock markets fell sharply on Monday.", "Safe.", "the cat sat on the mat and the dog sat too",
             "A cat was sitting on the mat while a dog sat as well.", "the the the the"]


def random_sentence(rng: random.Random) -> str:
    vocab = "the a cat dog sat on mat is safe drug for children kids and too was".split()
    return " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 14)))


@pytest.mark.parametrize("seed", range(30))
def test_bleu_matches_sacrebleu(seed):
    rng = random.Random(seed)
    cand = random_sentence(rng)
    refs = [random_sentence(rng) for _ in range(rng.randint(1, 3))]
    ours = bleu(words(cand), [words(r) for r in refs])
    theirs = BLEU(smooth_method="add-k", smooth_value=1, effective_order=False, tokenize="none").sentence_score(
        " ".join(words(cand)), [" ".join(words(r)) for r in refs]).score / 100
    assert ours == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("seed", range(30))
def test_rouge_matches_rouge_score(seed):
    rng = random.Random(seed)
    cand = random_sentence(rng)
    refs = [random_sentence(rng) for _ in range(rng.randint(1, 3))]
    scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False).score_multi(refs, cand)
    c, rs = words(cand), [words(r) for r in refs]
    assert max(rouge_n(c, r, 1) for r in rs) == pytest.approx(scores["rouge1"].fmeasure)
    assert max(rouge_n(c, r, 2) for r in rs) == pytest.approx(scores["rouge2"].fmeasure)
    assert max(rouge_l(c, r) for r in rs) == pytest.approx(scores["rougeL"].fmeasure)


def test_lcs_matches_the_textbook_table():
    rng = random.Random(0)
    for _ in range(200):
        a, b = random_sentence(rng).split(), random_sentence(rng).split()
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                table[i][j] = table[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(table[i - 1][j], table[i][j - 1])
        assert lcs(a, b) == table[-1][-1]


def test_bertscore_by_hand_matches_the_library():
    from bert_score import score

    cands, refs = SENTENCES[1:6], SENTENCES[:5]
    ours = BertScorer("distilroberta-base", 5)
    p, r, f = score(cands, refs, model_type="distilroberta-base", num_layers=5, idf=False, lang="en")
    for i, (c, ref) in enumerate(zip(cands, refs)):
        mine = ours.score(c, ref)
        assert mine == pytest.approx((p[i].item(), r[i].item(), f[i].item()), abs=1e-4)


def test_exact_match_and_token_f1():
    assert exact_match("The Paris.", "paris") == 1
    assert exact_match("Paris, France", "Paris") == 0
    assert token_f1("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert token_f1("It is not Paris", "Paris") == pytest.approx(0.4)
    assert token_f1("Lyon", "Paris") == 0


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert list(ranks([10, 20, 20, 30])) == [1, 2.5, 2.5, 4]
