"""Training smoke test: a few steps on tiny data must lower the loss + save artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import (
    METRICS_FILENAME,
    MODEL_CONFIG_FILENAME,
    TOKENIZER_FILENAME,
    WEIGHTS_FILENAME,
    GPTConfig,
    TrainConfig,
)
from src.data import TokenDataset
from src.model import GPT
from src.tokenizer import CharTokenizer
from src.train import configure_optimizer, cosine_lr, fit, perplexity

_SMOKE_CONFIG = TrainConfig(
    batch_size=16,
    max_steps=60,
    learning_rate=1e-2,
    warmup_steps=5,
    eval_interval=20,
    eval_iters=10,
    sample_interval=0,
)


def _tiny_setup() -> tuple[GPT, TokenDataset, CharTokenizer, GPTConfig]:
    torch.manual_seed(0)
    model_config = GPTConfig(
        vocab_size=17, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0
    )
    model = GPT(model_config)
    # Repeating pattern is learnable, so the loss should clearly drop.
    ids = torch.arange(model_config.vocab_size).repeat(64)
    dataset = TokenDataset(train_ids=ids, val_ids=ids.clone())
    tokenizer = CharTokenizer.train(
        "".join(chr(ord("a") + i) for i in range(model_config.vocab_size))
    )
    return model, dataset, tokenizer, model_config


def test_fit_lowers_loss_and_writes_artifacts(tmp_path: Path) -> None:
    model, dataset, tokenizer, model_config = _tiny_setup()
    metrics = fit(
        model,
        dataset,
        tokenizer,
        _SMOKE_CONFIG,
        model_config,
        device=torch.device("cpu"),
        models_dir=tmp_path,
        save_artifacts=True,
    )
    assert metrics["best_val_loss"] < metrics["initial_val_loss"]
    for filename in (WEIGHTS_FILENAME, MODEL_CONFIG_FILENAME, TOKENIZER_FILENAME, METRICS_FILENAME):
        assert (tmp_path / filename).exists()


def test_cosine_lr_warmup_and_decay() -> None:
    lr_max, lr_min, warmup, steps = 1.0, 0.1, 10, 100
    assert cosine_lr(0, warmup, steps, lr_max, lr_min) < lr_max  # warming up
    assert cosine_lr(warmup - 1, warmup, steps, lr_max, lr_min) == lr_max  # peak at end of warmup
    assert abs(cosine_lr(steps, warmup, steps, lr_max, lr_min) - lr_min) < 1e-9  # floor after end
    mid = cosine_lr(steps // 2, warmup, steps, lr_max, lr_min)
    assert lr_min < mid < lr_max


def test_configure_optimizer_splits_weight_decay() -> None:
    _, _, _, model_config = _tiny_setup()
    model = GPT(model_config)
    optimizer = configure_optimizer(model, weight_decay=0.1, learning_rate=1e-3)
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0


def test_perplexity_overflow_guard() -> None:
    assert perplexity(0.0) == 1.0
    assert perplexity(1000.0) == float("inf")
