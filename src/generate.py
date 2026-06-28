"""Autoregressive sampling: temperature, top-k, and top-p (nucleus) filtering.

These are pure functions over logits so the same sampling logic is shared by
``GPT.generate`` (used in training samples and the API) and the CLI added later.
All filtering masks logits to ``-inf`` *before* the softmax so removed tokens get
exactly zero probability.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.config import MODELS_DIR, GenerationConfig, resolve_device, seed_everything

if TYPE_CHECKING:
    from src.model import GPT


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the ``k`` highest-logit tokens per row; mask the rest to -inf."""
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability reaches ``p``.

    The highest-probability token is always kept, so ``p`` near 0 reduces to greedy.
    """
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Remove tokens once the cumulative prob *before* them already exceeds p.
    remove_sorted = (cumulative - probs) > p
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_index, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id from last-step logits.

    ``logits``: (B, vocab). Returns: (B, 1) LongTensor of sampled ids.
    ``temperature <= 0`` is treated as greedy (argmax) without dividing by zero.
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        logits = _top_k_filter(logits, top_k)
    if top_p is not None:
        logits = _top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def iter_generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """Yield one new token id (shape (B, 1)) at a time, autoregressively.

    The context is cropped to the model's block size each step. This is the single
    autoregressive loop shared by the model's one-shot ``generate`` and the API's
    streaming endpoint. The model is expected to already be in eval mode.
    """
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, next_id], dim=1)
        yield next_id


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = GenerationConfig()
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniGPT model.")
    parser.add_argument("--prompt", default="\n", help="Prompt to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k, help="0 disables top-k.")
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: load the trained model and print a completion for a prompt."""
    args = _build_arg_parser().parse_args(argv)
    if args.seed is not None:
        seed_everything(args.seed)
    device = resolve_device()

    # Imported here (not at module top) to avoid a circular import: model -> generate.
    from src.model import load_pretrained

    model, tokenizer = load_pretrained(args.models_dir, device)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
