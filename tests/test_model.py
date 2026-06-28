"""Model contract: forward shape, causal masking, block-size guard, generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import (
    MODEL_CONFIG_FILENAME,
    TOKENIZER_FILENAME,
    WEIGHTS_FILENAME,
    GPTConfig,
)
from src.model import GPT, load_pretrained, save_weights
from src.tokenizer import CharTokenizer


def test_forward_shape(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    batch, seq_len = 4, 5
    idx = torch.randint(0, tiny_config.vocab_size, (batch, seq_len))
    logits, loss = tiny_model(idx)
    assert logits.shape == (batch, seq_len, tiny_config.vocab_size)
    assert loss is None


def test_forward_returns_scalar_loss_with_targets(tiny_model: GPT, tiny_config: GPTConfig) -> None:
    batch, seq_len = 4, 5
    idx = torch.randint(0, tiny_config.vocab_size, (batch, seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (batch, seq_len))
    _, loss = tiny_model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0


def test_causal_mask_future_token_does_not_change_past_logits(
    tiny_model: GPT, tiny_config: GPTConfig
) -> None:
    """The #1 GPT bug: changing a future token must not alter earlier-position logits."""
    torch.manual_seed(1)
    seq_len = tiny_config.block_size
    idx = torch.randint(0, tiny_config.vocab_size, (1, seq_len))
    changed_at = seq_len // 2

    with torch.no_grad():
        logits_before, _ = tiny_model(idx)
        idx_future_changed = idx.clone()
        idx_future_changed[0, changed_at] = (idx[0, changed_at] + 1) % tiny_config.vocab_size
        logits_after, _ = tiny_model(idx_future_changed)

    # Positions before the edit cannot attend to it -> identical logits.
    assert torch.allclose(
        logits_before[:, :changed_at, :], logits_after[:, :changed_at, :], atol=1e-6
    )
    # Sanity: the edited position itself does change, so the test is not vacuous.
    assert not torch.allclose(
        logits_before[:, changed_at, :], logits_after[:, changed_at, :], atol=1e-6
    )


def test_forward_rejects_sequence_longer_than_block_size(
    tiny_model: GPT, tiny_config: GPTConfig
) -> None:
    idx = torch.zeros((1, tiny_config.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block_size"):
        tiny_model(idx)


def test_generate_produces_requested_length_and_respects_cropping(
    tiny_model: GPT, tiny_config: GPTConfig
) -> None:
    torch.manual_seed(2)
    batch = 2
    new_tokens = 5
    # Start at exactly block_size so generation must crop context every step.
    idx = torch.randint(0, tiny_config.vocab_size, (batch, tiny_config.block_size))
    out = tiny_model.generate(idx, max_new_tokens=new_tokens, temperature=1.0)
    assert out.shape == (batch, tiny_config.block_size + new_tokens)
    assert int(out.min()) >= 0
    assert int(out.max()) < tiny_config.vocab_size


def test_save_and_load_pretrained_reproduces_logits(tmp_path: Path) -> None:
    config = GPTConfig(vocab_size=17, block_size=8, n_layer=2, n_head=2, d_model=16, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(config).eval()
    config.save(tmp_path / MODEL_CONFIG_FILENAME)
    save_weights(model, tmp_path / WEIGHTS_FILENAME)
    CharTokenizer.train("abcdefghijklmnopq").save(tmp_path / TOKENIZER_FILENAME)

    loaded, tokenizer = load_pretrained(tmp_path, torch.device("cpu"))
    idx = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        original_logits, _ = model(idx)
        loaded_logits, _ = loaded(idx)
    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)
    assert tokenizer.vocab_size == config.vocab_size
