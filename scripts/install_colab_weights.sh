#!/usr/bin/env bash
# Copy Colab artifacts into models/ and verify they are the BPE TinyStories checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$HOME/Downloads}"
DST="$REPO_ROOT/models"

for f in model.safetensors config.json tokenizer.json metrics.json; do
  cp "$SRC/$f" "$DST/$f"
done

python3 "$REPO_ROOT/scripts/verify_colab_weights.py"
