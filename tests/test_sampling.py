"""Sampling behavior: greedy fallback, top-k/top-p filtering, determinism."""

from __future__ import annotations

import torch

from src.generate import _top_k_filter, _top_p_filter, sample_next_token

# Distinct logits so the argmax (index 1) is unambiguous.
LOGITS = torch.tensor([[1.0, 3.0, 2.0, 0.5, -1.0]])
ARGMAX = 1


def test_temperature_zero_is_greedy() -> None:
    out = sample_next_token(LOGITS, temperature=0.0, top_k=None, top_p=None)
    assert out.shape == (1, 1)
    assert int(out) == ARGMAX


def test_negative_temperature_is_greedy() -> None:
    out = sample_next_token(LOGITS, temperature=-1.0, top_k=None, top_p=None)
    assert int(out) == ARGMAX


def test_top_k_one_always_returns_argmax() -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        out = sample_next_token(LOGITS, temperature=1.0, top_k=1, top_p=None, generator=generator)
        assert int(out) == ARGMAX


def test_top_p_zero_keeps_only_top_token() -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        out = sample_next_token(LOGITS, temperature=1.0, top_k=None, top_p=0.0, generator=generator)
        assert int(out) == ARGMAX


def test_top_k_filter_keeps_exactly_k_tokens() -> None:
    filtered = _top_k_filter(LOGITS.clone(), 2)
    finite = torch.isfinite(filtered)
    assert int(finite.sum()) == 2
    assert bool(finite[0, 1]) and bool(finite[0, 2])


def test_top_p_filter_always_keeps_top_token() -> None:
    filtered = _top_p_filter(LOGITS.clone(), 0.0)
    assert int(torch.isfinite(filtered).sum()) >= 1
    assert int(torch.argmax(filtered)) == ARGMAX


def test_sample_shape_dtype_and_range() -> None:
    logits = torch.randn(4, 11)
    out = sample_next_token(logits, temperature=1.0, top_k=5, top_p=0.9)
    assert out.shape == (4, 1)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0
    assert int(out.max()) < 11


def test_generator_makes_sampling_reproducible() -> None:
    logits = torch.randn(2, 7)
    first = sample_next_token(logits, 1.0, None, None, generator=torch.Generator().manual_seed(123))
    second = sample_next_token(
        logits, 1.0, None, None, generator=torch.Generator().manual_seed(123)
    )
    assert torch.equal(first, second)
