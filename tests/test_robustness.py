"""The perturbations change what they claim to change, and nothing else."""

import random

import pytest

from robustness import (NEAR, PERTURBATIONS, first_word, keyboard_typos, number_words, numbers_as_words,
                        perturbed_prompt, sign_test, whitespace)

TEXT = "Is the 1996 World Series the 92nd edition of the championship? It had 7 games in 9 days."


def test_number_words():
    assert [number_words(n) for n in (0, 7, 13, 40, 92, 100, 305, 1996, 2000)] == [
        "zero", "seven", "thirteen", "forty", "ninety-two", "one hundred", "three hundred and five",
        "one thousand nine hundred and ninety-six", "two thousand"]
    assert numbers_as_words("7 games in 1996, the 92nd") == "seven games in one thousand nine hundred and ninety-six, the 92nd"


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_perturbations_repeat_with_the_same_seed(name):
    f = PERTURBATIONS[name]
    assert f(TEXT, random.Random(1)) == f(TEXT, random.Random(1))


def test_typos_only_swap_letters_for_neighbouring_keys():
    out = keyboard_typos(TEXT * 20, random.Random(0), rate=0.3)
    assert len(out) == len(TEXT * 20)
    changed = [(a, b) for a, b in zip(TEXT * 20, out) if a != b]
    assert changed and all(b.lower() in NEAR[a.lower()] for a, b in changed)


def test_whitespace_keeps_every_other_character():
    out = whitespace(TEXT * 20, random.Random(0))
    assert out.replace(" ", "") == (TEXT * 20).replace(" ", "")


def test_upper_and_lower_case_only_change_case():
    for name in ("all lower case", "random upper case"):
        assert PERTURBATIONS[name](TEXT, random.Random(0)).lower() == TEXT.lower()


def test_first_word_reads_yes_and_no():
    assert first_word("Yes.") == "yes" and first_word(" No, it is not") == "no" and first_word("") == ""


def test_sign_test():
    assert sign_test(0, 5) == 2 / 32  # all five the same way: 2 of 32 equally likely splits
    assert sign_test(2, 7) == sign_test(7, 2) == pytest.approx(2 * 46 / 512)
    assert sign_test(3, 3) == sign_test(0, 0) == 1.0


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_the_instruction_is_never_rewritten(name):
    item = {"passage": TEXT * 5, "question": "is the 1996 world series the 92nd edition?", "answer": "yes"}
    out = perturbed_prompt(item, PERTURBATIONS[name], random.Random(0))
    assert "\n\nQuestion: " in out and out.endswith("\nAnswer yes or no.")
