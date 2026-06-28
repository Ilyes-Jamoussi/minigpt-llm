"""Verify that models/ contains the Colab BPE + TinyStories checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "model.safetensors"
METRICS_PATH = MODELS_DIR / "metrics.json"
CONFIG_PATH = MODELS_DIR / "config.json"

MIN_WEIGHTS_BYTES = 40_000_000


def main() -> int:
    errors: list[str] = []

    if not WEIGHTS_PATH.exists():
        errors.append(f"missing {WEIGHTS_PATH.name}")
    else:
        size = WEIGHTS_PATH.stat().st_size
        if size < MIN_WEIGHTS_BYTES:
            errors.append(
                f"{WEIGHTS_PATH.name} is {size:,} bytes (~{size / 1e6:.1f} MB); "
                f"expected >= {MIN_WEIGHTS_BYTES / 1e6:.0f} MB for the ~14M-param BPE model"
            )

    if not METRICS_PATH.exists():
        errors.append(f"missing {METRICS_PATH.name}")
    else:
        metrics = json.loads(METRICS_PATH.read_text())
        if metrics.get("dataset") != "tinystories":
            errors.append(f"metrics dataset={metrics.get('dataset')!r}, expected 'tinystories'")
        if metrics.get("tokenizer_type") != "bpe":
            tok = metrics.get("tokenizer_type")
            errors.append(f"metrics tokenizer_type={tok!r}, expected 'bpe'")
        ppl = metrics.get("val_perplexity")
        if not isinstance(ppl, (int, float)) or ppl > 6.5:
            errors.append(f"metrics val_perplexity={ppl!r}, expected <= 6.5 after Colab training")

    if not CONFIG_PATH.exists():
        errors.append(f"missing {CONFIG_PATH.name}")
    else:
        config = json.loads(CONFIG_PATH.read_text())
        if config.get("vocab_size", 0) < 1000:
            errors.append(f"config vocab_size={config.get('vocab_size')}, expected ~8192")

    if errors:
        print("Colab weights check FAILED:")
        for err in errors:
            print(f"  - {err}")
        print("\nRe-download the four files from Colab: repo/models/")
        print("Open metrics.json first — it must show tinystories + bpe.")
        return 1

    print("Colab weights check OK — ready to commit and deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
