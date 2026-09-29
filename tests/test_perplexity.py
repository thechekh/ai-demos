"""Surprise and perplexity: the definitions hold, and the numbers match the library's own loss."""

import math

import pytest
import torch

from perplexity import START, bits_from_logits, perplexity, text_bits


def test_a_uniform_guess_over_v_tokens_has_perplexity_v():
    vocab = 1_000
    logits = torch.zeros(5, vocab)  # every token equally likely
    bits = bits_from_logits(logits, torch.tensor([3, 14, 159, 265, 358]))
    assert bits.tolist() == pytest.approx([math.log2(vocab)] * 5)
    assert perplexity(bits.tolist()) == pytest.approx(vocab)


def test_a_certain_guess_has_zero_surprise():
    logits = torch.full((1, 10), -1e4)
    logits[0, 7] = 0
    assert bits_from_logits(logits, torch.tensor([7])).item() == pytest.approx(0, abs=1e-6)


def test_the_bits_match_the_models_own_loss():
    from llm import load_model, load_tokenizer

    tokenizer, model = load_tokenizer(), load_model()
    text = "The Eiffel Tower is in Paris, and it was finished in 1889."
    bits = text_bits(model, tokenizer, text)
    ids = tokenizer.encode(START + text, add_special_tokens=False)
    with torch.no_grad():
        loss = model(torch.tensor([ids]), labels=torch.tensor([ids])).loss.item()  # mean nats, all tokens
    assert sum(bits) / len(bits) * math.log(2) == pytest.approx(loss, rel=1e-4)
