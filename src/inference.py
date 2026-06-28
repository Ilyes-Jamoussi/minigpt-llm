"""Load the trained model and stream text completions (used by Streamlit and tests)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import METRICS_FILENAME, MODELS_DIR, resolve_device
from src.generate import iter_generate
from src.model import GPT, load_pretrained
from src.tokenizer import Tokenizer


@dataclass(frozen=True)
class LoadedModel:
    """Trained GPT, tokenizer, and device — loaded once at startup."""

    model: GPT
    tokenizer: Tokenizer
    device: torch.device


def load_for_inference(models_dir: Path = MODELS_DIR) -> LoadedModel:
    """Load model weights and tokenizer from ``models_dir`` onto the best device."""
    device = resolve_device()
    model, tokenizer = load_pretrained(models_dir, device)
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def load_metrics(models_dir: Path = MODELS_DIR) -> dict[str, object]:
    """Read training metrics written by ``train.py``."""
    raw: object = json.loads((models_dir / METRICS_FILENAME).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"invalid metrics file: {models_dir / METRICS_FILENAME}")
    return raw


def stream_completion(
    loaded: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Iterator[str]:
    """Yield decoded text chunks as tokens are sampled autoregressively."""
    prompt_ids = loaded.tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.device)
    generated: list[int] = []
    emitted = ""
    for next_id in iter_generate(loaded.model, idx, max_new_tokens, temperature, top_k, top_p):
        generated.append(int(next_id))
        text = loaded.tokenizer.decode(generated)
        delta = text[len(emitted) :]
        emitted = text
        if delta:
            yield delta
