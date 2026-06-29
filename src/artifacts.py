"""Fetch committed model artifacts when Git LFS pointers or stale weights are present.

Streamlit Community Cloud clones the repo without Git LFS, leaving 133-byte pointer
files instead of the ~57 MB weight tensor. This module re-downloads the real artifacts
from GitHub's media CDN when the on-disk copy is missing or invalid.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from src.config import (
    METRICS_FILENAME,
    MODEL_CONFIG_FILENAME,
    MODELS_DIR,
    TOKENIZER_FILENAME,
    WEIGHTS_FILENAME,
)

logger = logging.getLogger(__name__)

_GITHUB_MEDIA_BASE = (
    "https://media.githubusercontent.com/media/Ilyes-Jamoussi/minigpt-llm/main/models"
)
_MIN_WEIGHTS_BYTES = 40_000_000
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_ARTIFACT_FILES = (
    WEIGHTS_FILENAME,
    MODEL_CONFIG_FILENAME,
    TOKENIZER_FILENAME,
    METRICS_FILENAME,
)


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    return path.read_bytes().startswith(_LFS_POINTER_PREFIX)


def _metrics_show_bpe_model(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    metrics = json.loads(metrics_path.read_text())
    dataset = metrics.get("dataset")
    tokenizer_type = metrics.get("tokenizer_type")
    return bool(dataset == "tinystories" and tokenizer_type == "bpe")


def artifacts_ready(models_dir: Path) -> bool:
    """Return True when weights and metrics look like the committed BPE checkpoint."""
    weights = models_dir / WEIGHTS_FILENAME
    metrics = models_dir / METRICS_FILENAME
    if not weights.exists() or not metrics.exists():
        return False
    if _is_lfs_pointer(weights) or weights.stat().st_size < _MIN_WEIGHTS_BYTES:
        return False
    return _metrics_show_bpe_model(metrics)


def _should_bootstrap(models_dir: Path) -> bool:
    if os.environ.get("MINIGPT_SKIP_REMOTE_ARTIFACTS") == "1":
        return False
    if artifacts_ready(models_dir):
        return False
    # Only auto-fetch for the default repo models/ dir (Streamlit / fresh clones).
    return models_dir.resolve() == MODELS_DIR.resolve()


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("downloading %s", dest.name)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_model_artifacts(models_dir: Path = MODELS_DIR) -> bool:
    """Download artifacts from GitHub when needed. Returns True if a download ran."""
    if not _should_bootstrap(models_dir):
        return False

    for name in _ARTIFACT_FILES:
        url = f"{_GITHUB_MEDIA_BASE}/{name}"
        _download_file(url, models_dir / name)

    if not artifacts_ready(models_dir):
        raise RuntimeError("model artifacts download completed but validation failed")

    logger.info("model artifacts ready in %s", models_dir)
    return True
