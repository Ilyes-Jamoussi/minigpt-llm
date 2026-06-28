"""Shared pytest fixtures: a tiny, fast model so tests stay quick and deterministic."""

from __future__ import annotations

import pytest
import torch

from src.config import GPTConfig
from src.model import GPT


@pytest.fixture
def tiny_config() -> GPTConfig:
    # Dropout 0 so forward passes are deterministic in eval mode.
    return GPTConfig(vocab_size=17, block_size=8, n_layer=2, n_head=2, d_model=16, dropout=0.0)


@pytest.fixture
def tiny_model(tiny_config: GPTConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_config).eval()
