"""The sampler follows its rules, and the hand-written loop agrees with the library's greedy decoding."""

import pytest
import torch

from sampling import first_difference, next_token, probabilities, sample

LOGITS = torch.tensor([[2.0, 1.0, 0.0, -1.0]])


def test_temperature_zero_is_the_most_likely_token():
    assert next_token(LOGITS, 0.0, 1.0, torch.Generator().manual_seed(0)).item() == 0


def test_lower_temperature_sharpens_and_higher_flattens():
    cold, warm, hot = (probabilities(LOGITS, t)[0] for t in (0.5, 1.0, 2.0))
    for p in (cold, warm, hot):
        assert abs(float(p.sum()) - 1) < 1e-6
    assert cold[0] > warm[0] > hot[0] and cold[-1] < warm[-1] < hot[-1]


def test_sampling_follows_the_probabilities():
    generator = torch.Generator().manual_seed(0)
    draws = next_token(LOGITS.repeat(40_000, 1), 1.0, 1.0, generator)
    freq = torch.bincount(draws, minlength=4).float() / len(draws)
    assert torch.allclose(freq, probabilities(LOGITS, 1.0)[0], atol=0.01)


def test_top_p_keeps_the_smallest_set_that_reaches_it():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    draws = next_token(logits.repeat(5_000, 1), 1.0, 0.6, torch.Generator().manual_seed(1))
    assert set(draws.tolist()) == {0, 1}  # 0.5 alone is short of 0.6; 0.5 + 0.3 reaches it
    tiny = next_token(logits.repeat(100, 1), 1.0, 1e-9, torch.Generator().manual_seed(2))
    assert set(tiny.tolist()) == {0}


def test_first_difference():
    assert first_difference([1, 2, 3], [1, 2, 3]) is None
    assert first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert first_difference([1, 2], [1, 2, 3]) == 2


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from llm import load_model, load_tokenizer

    return load_model(), load_tokenizer()


def test_the_hand_written_loop_matches_the_library_and_seeds_repeat(model_and_tokenizer):
    from llm import greedy

    model, tokenizer = model_and_tokenizer
    prompt = "Write one sentence about the sea."
    ours = sample(model, tokenizer, prompt, 1, 16, 0.0)[0]
    assert ours == tokenizer.decode(greedy(model, tokenizer, [prompt], 16)[0])
    assert sample(model, tokenizer, prompt, 3, 8, 1.0, seed=5) == sample(model, tokenizer, prompt, 3, 8, 1.0, seed=5)
