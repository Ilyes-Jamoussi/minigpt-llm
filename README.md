# MiniGPT

[![CI](https://github.com/Ilyes-Jamoussi/minigpt-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Ilyes-Jamoussi/minigpt-llm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A GPT-style decoder-only language model **built from scratch in PyTorch** — causal multi-head
attention, learned positional embeddings, weight tying, and autoregressive sampling — served
through a **typed, tested FastAPI service** that **streams tokens live** to a web demo.

No `nn.Transformer`, no HuggingFace `transformers` for the model. The attention, causal
masking, byte-level BPE tokenizer, and sampling are all implemented here.

### [Live Demo](https://minigpt-llm.streamlit.app)

![MiniGPT streaming demo](docs/demo.png)

## How It Works

Text goes through the following pipeline:

1. **Tokenization** — a from-scratch character-level or byte-level BPE tokenizer maps text to
   integer ids. The BPE tokenizer is trained on the train split only.
2. **GPT forward pass** — token + positional embeddings pass through pre-LN transformer blocks
   with causal multi-head self-attention (position `t` attends only to `≤ t`), then a tied LM
   head outputs next-token logits.
3. **Training** — next-token cross-entropy with AdamW, cosine LR schedule with linear warmup,
   gradient clipping, and mixed precision on CUDA.
4. **Generation** — autoregressive sampling with temperature, top-k, and top-p (nucleus);
   greedy when `temperature → 0`.
5. **Serving** — FastAPI loads the model once at startup and exposes full and streaming
   (`Server-Sent Events`) generation endpoints plus a minimal web demo.

## Model Architecture

```
Prompt text
  → Tokenization (byte-level BPE, vocab 8192)
  → Token Embedding (d=384) + Positional Embedding
  → Transformer Block × 6
      ├── Causal Multi-Head Self-Attention (6 heads, lower-triangular mask)
      ├── Residual + Layer Norm (pre-LN)
      ├── MLP (384 → 1536 → 384, GELU)
      └── Residual + Layer Norm (pre-LN)
  → Final LayerNorm
  → Linear LM head (weights tied to token embedding)
  → Autoregressive sampling → generated text
```

Committed model (TinyStories, byte-level BPE): **~13.9M parameters**, `block_size=256`.

## Dataset

The committed model is trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(simple English stories for children). The train/val split is deterministic and contiguous (90/10);
the byte-level BPE tokenizer (vocab 8192) is trained on the train split only.

A smaller character-level TinyShakespeare configuration remains available for quick local smoke tests
(`python -m src.train --dataset tinyshakespeare --tokenizer-type char --max-steps 1000`).

## Training

Run on a GPU via the Colab notebook (~5–9 hours for the committed model on a T4).

- Optimizer: AdamW (lr=3e-4 → 3e-5 cosine, warmup 300 steps)
- Weight decay: 0.1 (matrices only), gradient clip: 1.0
- Mixed precision (AMP) when CUDA is available
- Best checkpoint selected by validation loss
- Seeded (`random` / `numpy` / `torch`) for reproducible runs
- Artifacts saved as `safetensors` + JSON (no pickle)

## Results

Committed BPE model (TinyStories, 30,000 training steps, Colab T4 GPU):

| Metric | Value |
| --- | --- |
| Parameters | ~13,900,000 |
| Tokenizer | byte-level BPE, vocab 8192 |
| Dataset | TinyStories (~150M characters) |
| Validation perplexity | **5.32** |
| Tokens seen | **~246,000,000** |

Sample generation (`temperature=0.8, top_k=50, top_p=0.95`, prompt `Once upon a time`):

```
Once upon a time, there was a little girl named Lily. She loved to play with her toys and her
best friend, a teddy bear named Tim. One day, Lily and Tim went to the park. They saw a big,
shiny slide. Lily wanted to go on the slide, but Tim was scared.
```

The model produces coherent short-story English after training on TinyStories with BPE tokenization.
It is not state-of-the-art — it demonstrates the full from-scratch pipeline at portfolio scale.

## Limitations

The committed model is a ~14M-parameter decoder trained on a subset of TinyStories. Quality is
appropriate for a portfolio demo (coherent sentences, simple narratives), not production chat.
Scaling context length, model size, or training budget would be the natural next steps.

## Project Structure

```
├── app.py                     # Streamlit UI (rendering only)
├── assets/
│   └── styles.css             # UI styling (kept out of logic)
├── api/
│   └── main.py                # FastAPI service (SSE streaming)
├── demo/
│   ├── index.html             # Streaming web UI
│   ├── app.js                 # fetch + SSE client
│   └── styles.css             # UI styling (kept out of logic)
├── src/
│   ├── config.py              # Single source of truth: hyperparameters, paths, seed
│   ├── tokenizer.py           # Char + byte-level BPE tokenizers (from scratch)
│   ├── model.py               # GPT decoder-only model (from scratch)
│   ├── data.py                # Dataset download, split, batching
│   ├── train.py               # Training loop, checkpointing, evaluation
│   ├── generate.py            # Sampling utils + CLI
│   └── inference.py           # Model loading + streaming (used by Streamlit)
├── models/
│   ├── model.safetensors      # Trained weights (Git LFS)
│   ├── config.json            # Model hyperparameters
│   ├── tokenizer.json         # Fitted tokenizer
│   └── metrics.json           # Training metrics
├── tests/                     # Tokenizer, causal mask, model, train, API
├── notebooks/
│   └── train_minigpt.ipynb    # Colab notebook for BPE + TinyStories
├── Dockerfile                 # Minimal CPU container for API + demo
├── pyproject.toml             # ruff / mypy / pytest config
├── requirements.txt           # Runtime dependencies (pinned)
└── requirements-dev.txt       # Dev/tooling dependencies (pinned)
```

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), connected
directly to this GitHub repository. Model weights are stored via Git LFS. Any push to
`main` triggers an automatic redeployment.

The repo also ships a minimal CPU `Dockerfile` for self-hosting the FastAPI service + demo:

```bash
docker build -t minigpt .
docker run -p 8000:8000 minigpt
```

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py          # live streaming demo (same as deployed app)
uvicorn api.main:app          # FastAPI + browser demo at http://localhost:8000/
```

### Train on Kaggle (recommended if Colab GPU quota is exhausted)

Kaggle offers free GPU hours and supports **unattended training** via **Save & Run All (Commit)** —
the notebook runs in the cloud even if you close the browser. Artifacts are written to
`/kaggle/working/minigpt-artifacts/` and appear in the notebook **Output** tab.

1. Create a free [Kaggle](https://www.kaggle.com) account (phone verification required).
2. **New Notebook** → upload `notebooks/train_minigpt_kaggle.ipynb` from this repo
   (or copy the cells).
3. Notebook settings (right panel): **Accelerator → GPU T4 x2**, **Internet → On**.
4. Run the **Setup** cell once — confirm `cuda available: True`.
5. **Save Version → Save & Run All (Commit)** — training runs alone (~5–9 h).
6. When the commit status is **Complete**, open **Output** → download `minigpt-artifacts/`
   (four files).
7. Install locally:

```bash
bash scripts/install_colab_weights.sh ~/Downloads/minigpt-artifacts
```

### Train on Google Colab

1. Open `notebooks/train_minigpt.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime to GPU (Runtime → Change runtime type → GPU).
3. Run all cells — the notebook downloads TinyStories, trains the BPE tokenizer and model,
   and writes artifacts to `models/`.
4. Download the four files from `models/` **immediately** before the session ends.

### Install Colab weights

After training in Colab, download the four files from `repo/models/` and install them locally:

```bash
bash scripts/install_colab_weights.sh ~/Downloads
python3 scripts/verify_colab_weights.py   # must print OK (~56 MB weights, tinystories + bpe)
```

Then commit and push — Streamlit Cloud redeploys automatically from `main`.

### Quick local smoke train (char-level)

```bash
python -m src.train --dataset tinyshakespeare --tokenizer-type char --max-steps 1000
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200
```

## Development

```bash
pip install -r requirements-dev.txt
ruff format . && ruff check . && mypy src api && pytest
```

## Tech Stack

- **PyTorch** — tensors, autograd, model building
- **safetensors** — safe model-weight serialization
- **FastAPI + uvicorn** — typed streaming inference API
- **datasets** — TinyStories loader (training only)
- **ruff / mypy / pytest** — linting, type checking, tests
