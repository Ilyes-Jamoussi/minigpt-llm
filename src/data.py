"""Dataset download, deterministic train/val split, tokenization, and batching.

Kept separate from the training loop so each concern stays small. The tokenizer
is trained on the train split only (no validation leakage), per CONVENTIONS s8.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.config import DATA_DIR, TrainConfig
from src.tokenizer import Tokenizer, build_tokenizer

logger = logging.getLogger(__name__)

Split = Literal["train", "val"]

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINYSTORIES_DATASET = "roneneldan/TinyStories"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)


def _load_tinyshakespeare() -> str:
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        _download(TINYSHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def _load_tinystories(max_chars: int | None) -> str:
    # datasets is a training-only dependency, imported lazily so serving stays light.
    from datasets import load_dataset

    logger.info("loading %s (train split)", TINYSTORIES_DATASET)
    dataset = load_dataset(TINYSTORIES_DATASET, split="train")
    parts: list[str] = []
    total = 0
    for row in dataset:
        story = row["text"]
        parts.append(story)
        total += len(story) + 2
        if max_chars is not None and total >= max_chars:
            break
    return "\n\n".join(parts)


def load_text(config: TrainConfig) -> str:
    """Return the raw corpus text for the configured dataset, downloading once.

    Applies ``config.max_chars`` so large datasets can be trained on a budget.
    """
    if config.dataset == "tinyshakespeare":
        text = _load_tinyshakespeare()
    elif config.dataset == "tinystories":
        text = _load_tinystories(config.max_chars)
    else:
        raise ValueError(f"unknown dataset: {config.dataset!r}")

    if config.max_chars is not None:
        text = text[: config.max_chars]
    return text


def train_val_split(text: str, val_fraction: float) -> tuple[str, str]:
    """Split contiguously so the split is deterministic and leakage-free."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    split_at = int(len(text) * (1.0 - val_fraction))
    return text[:split_at], text[split_at:]


@dataclass
class TokenDataset:
    """Train/val token id streams plus random contiguous batch sampling.

    ``train_ids``/``val_ids``: 1-D LongTensors of token ids.
    """

    train_ids: torch.Tensor
    val_ids: torch.Tensor

    def get_batch(
        self,
        split: Split,
        batch_size: int,
        block_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of input/target sequences, each shifted by one token.

        Returns ``(x, y)`` with shape ``(batch_size, block_size)``; ``y`` is ``x``
        advanced by one position (next-token targets).
        """
        data = self.train_ids if split == "train" else self.val_ids
        high = len(data) - block_size
        if high <= 0:
            raise ValueError(
                f"{split} split has {len(data)} tokens, too few for block_size={block_size}"
            )
        indices = torch.randint(high, (batch_size,), generator=generator).tolist()
        x = torch.stack([data[i : i + block_size] for i in indices])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in indices])
        return x.to(device), y.to(device)


def prepare_data(config: TrainConfig) -> tuple[Tokenizer, TokenDataset]:
    """Download, split, train the tokenizer on train, and encode both splits."""
    text = load_text(config)
    train_text, val_text = train_val_split(text, config.val_fraction)
    tokenizer = build_tokenizer(config.tokenizer_type, train_text, config.bpe_vocab_size)
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    logger.info(
        "Prepared %s: %d train / %d val tokens, vocab_size=%d",
        config.dataset,
        train_ids.numel(),
        val_ids.numel(),
        tokenizer.vocab_size,
    )
    return tokenizer, TokenDataset(train_ids, val_ids)
