"""Single source of truth: hyperparameters, paths, seeding, and device resolution.

Every other module imports its constants from here. The model architecture
(``GPTConfig``) is persisted next to the weights as ``config.json`` so that
training, generation, and serving all reconstruct the exact same model.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

WEIGHTS_FILENAME = "model.safetensors"
MODEL_CONFIG_FILENAME = "config.json"
TOKENIZER_FILENAME = "tokenizer.json"
METRICS_FILENAME = "metrics.json"

DEFAULT_SEED = 1337


@dataclass(frozen=True)
class GPTConfig:
    """Decoder-only GPT architecture. Persisted as ``config.json``.

    These dims are the single constants shared by the model, training,
    generation, and serving code.
    """

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> GPTConfig:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop hyperparameters (data, optimizer, schedule, logging).

    ``block_size`` is intentionally absent: the sequence length lives only in
    ``GPTConfig`` to avoid duplicating that constant.
    """

    dataset: str = "tinyshakespeare"
    tokenizer_type: str = "char"
    bpe_vocab_size: int = 8192
    val_fraction: float = 0.1
    # Cap the corpus to a character budget (None = use all). Used to keep large
    # datasets like TinyStories within a tractable token budget on one GPU.
    max_chars: int | None = None

    batch_size: int = 64
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    eval_interval: int = 250
    eval_iters: int = 200
    sample_interval: int = 1000
    sample_prompt: str = "\n"
    sample_max_new_tokens: int = 200

    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class GenerationConfig:
    """Default sampling parameters. The API may override these per request."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95


def resolve_device() -> torch.device:
    """Pick the best available device once: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
