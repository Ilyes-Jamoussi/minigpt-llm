"""Reproducible training loop: AdamW + cosine schedule, AMP on CUDA, checkpointing.

The loop is split so it stays testable: ``cosine_lr``, ``configure_optimizer``,
``evaluate``, and ``fit`` are pure-ish and network-free, while ``train`` wires in
data download/preparation and ``main`` is the CLI.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from src.config import (
    METRICS_FILENAME,
    MODEL_CONFIG_FILENAME,
    MODELS_DIR,
    TOKENIZER_FILENAME,
    WEIGHTS_FILENAME,
    GenerationConfig,
    GPTConfig,
    TrainConfig,
    resolve_device,
    seed_everything,
)
from src.data import TokenDataset, prepare_data
from src.model import GPT, save_weights
from src.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Above this loss, perplexity (exp(loss)) overflows to a meaningless number.
_MAX_LOSS_FOR_PERPLEXITY = 20.0


def perplexity(loss: float) -> float:
    """Convert a cross-entropy loss to perplexity, guarding against overflow."""
    return math.exp(loss) if loss < _MAX_LOSS_FOR_PERPLEXITY else math.inf


def cosine_lr(
    step: int, warmup_steps: int, max_steps: int, learning_rate: float, min_lr: float
) -> float:
    """Linear warmup for ``warmup_steps``, then cosine decay to ``min_lr``."""
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizer(
    model: nn.Module, weight_decay: float, learning_rate: float
) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only (not biases/LayerNorm gains)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: TokenDataset,
    config: TrainConfig,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Average the loss over ``eval_iters`` batches for the train and val splits."""
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = dataset.get_batch(split, config.batch_size, block_size, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("model returned no loss during evaluation")
            losses[i] = loss.item()
        out[split] = float(losses.mean())
    if was_training:
        model.train()
    return out


def _log_sample(
    model: GPT, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, device: torch.device
) -> None:
    defaults = GenerationConfig()
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=defaults.temperature,
        top_k=defaults.top_k,
        top_p=defaults.top_p,
    )
    logger.info("sample generation:\n%s", tokenizer.decode(output[0].tolist()))


def _save_checkpoint(
    model: GPT,
    tokenizer: Tokenizer,
    model_config: GPTConfig,
    train_config: TrainConfig,
    eval_losses: dict[str, float],
    step: int,
    tokens_seen: int,
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    save_weights(model, models_dir / WEIGHTS_FILENAME)
    model_config.save(models_dir / MODEL_CONFIG_FILENAME)
    tokenizer.save(models_dir / TOKENIZER_FILENAME)
    metrics: dict[str, object] = {
        "dataset": train_config.dataset,
        "tokenizer_type": train_config.tokenizer_type,
        "num_params": model.num_parameters(),
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": eval_losses["train"],
        "val_loss": eval_losses["val"],
        "val_perplexity": perplexity(eval_losses["val"]),
        "model_config": asdict(model_config),
    }
    (models_dir / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2) + "\n")


def fit(
    model: GPT,
    dataset: TokenDataset,
    tokenizer: Tokenizer,
    train_config: TrainConfig,
    model_config: GPTConfig,
    device: torch.device,
    models_dir: Path = MODELS_DIR,
    save_artifacts: bool = True,
) -> dict[str, float]:
    """Run the training loop, checkpointing the best model by validation loss.

    Returns a small metrics dict (initial/best val loss and perplexity) so callers
    and tests can verify the loss actually went down.
    """
    block_size = model_config.block_size
    optimizer = configure_optimizer(model, train_config.weight_decay, train_config.learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_loss = math.inf
    initial_val_loss = math.inf
    start = time.time()
    model.train()

    for step in range(train_config.max_steps):
        is_last = step == train_config.max_steps - 1
        if step % train_config.eval_interval == 0 or is_last:
            losses = evaluate(model, dataset, train_config, block_size, device)
            if math.isinf(initial_val_loss):
                initial_val_loss = losses["val"]
            logger.info(
                "step %d/%d | train %.4f | val %.4f | val ppl %.2f",
                step,
                train_config.max_steps,
                losses["train"],
                losses["val"],
                perplexity(losses["val"]),
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if save_artifacts:
                    tokens_seen = step * train_config.batch_size * block_size
                    _save_checkpoint(
                        model,
                        tokenizer,
                        model_config,
                        train_config,
                        losses,
                        step,
                        tokens_seen,
                        models_dir,
                    )

        if train_config.sample_interval and step > 0 and step % train_config.sample_interval == 0:
            _log_sample(
                model,
                tokenizer,
                train_config.sample_prompt,
                train_config.sample_max_new_tokens,
                device,
            )

        lr = cosine_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.learning_rate,
            train_config.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = dataset.get_batch("train", train_config.batch_size, block_size, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("model returned no loss during training")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    logger.info(
        "training finished in %.1fs | best val loss %.4f | best val ppl %.2f",
        time.time() - start,
        best_val_loss,
        perplexity(best_val_loss),
    )
    return {
        "initial_val_loss": initial_val_loss,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": perplexity(best_val_loss),
    }


def train(
    train_config: TrainConfig | None = None,
    model_config: GPTConfig | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, float]:
    """Prepare data, build the model, and train it end to end.

    ``model_config`` is optional: its ``vocab_size`` is always taken from the
    trained tokenizer so the two can never disagree.
    """
    train_config = train_config or TrainConfig()
    seed_everything(train_config.seed)
    device = resolve_device()
    logger.info(
        "device: %s | dataset: %s | tokenizer: %s",
        device,
        train_config.dataset,
        train_config.tokenizer_type,
    )

    tokenizer, dataset = prepare_data(train_config)
    if model_config is None:
        model_config = GPTConfig(vocab_size=tokenizer.vocab_size)
    elif model_config.vocab_size != tokenizer.vocab_size:
        logger.warning(
            "overriding model vocab_size %d -> %d to match tokenizer",
            model_config.vocab_size,
            tokenizer.vocab_size,
        )
        model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    model = GPT(model_config).to(device)
    logger.info("model parameters: %.2fM", model.num_parameters() / 1e6)
    return fit(model, dataset, tokenizer, train_config, model_config, device, models_dir)


def main(argv: list[str] | None = None) -> None:
    """CLI: train with optional overrides of the most common hyperparameters."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train MiniGPT.")
    parser.add_argument("--dataset", default=defaults.dataset)
    parser.add_argument("--tokenizer-type", default=defaults.tokenizer_type)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    train_config = replace(
        defaults,
        dataset=args.dataset,
        tokenizer_type=args.tokenizer_type,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )
    train(train_config, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
