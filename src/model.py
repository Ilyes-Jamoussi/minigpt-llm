"""From-scratch decoder-only GPT in PyTorch (no nn.Transformer, no HF).

Tensor shape conventions used throughout:
    B = batch size, T = sequence length, C = d_model, V = vocab_size.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_model as _safetensors_load_model
from safetensors.torch import save_model as _safetensors_save_model
from torch import nn

from src.config import MODEL_CONFIG_FILENAME, TOKENIZER_FILENAME, WEIGHTS_FILENAME, GPTConfig
from src.generate import iter_generate
from src.tokenizer import Tokenizer, load_tokenizer


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (position t attends to <= t)."""

    # Declared for type checkers; set at runtime by ``register_buffer``.
    causal_mask: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)."""
        batch, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = attn @ v  # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        projected: torch.Tensor = self.resid_dropout(self.c_proj(out))
        return projected


class MLP(nn.Module):
    """Position-wise feed-forward: Linear -> GELU -> Linear, with dropout."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.proj = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.dropout(self.proj(F.gelu(self.fc(x))))
        return out


class Block(nn.Module):
    """Pre-LN transformer block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        out: torch.Tensor = x + self.mlp(self.ln2(x))
        return out


class GPT(nn.Module):
    """Decoder-only GPT with learned positional embeddings and a tied LM head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: the output projection shares the token-embedding matrix.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Total trainable parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids -> logits: (B, T, V); loss is set when targets given.

        ``targets``: (B, T) next-token ids. Returns ``(logits, loss)`` where loss is
        the mean cross-entropy over all positions, or ``None`` for inference.
        """
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(seq_len, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` (B, T) by ``max_new_tokens`` tokens.

        The context is cropped to the last ``block_size`` tokens each step so the
        model never sees a sequence longer than it was built for.
        """
        was_training = self.training
        self.eval()
        pieces = [idx]
        pieces.extend(
            iter_generate(self, idx, max_new_tokens, temperature, top_k, top_p, generator)
        )
        if was_training:
            self.train()
        return torch.cat(pieces, dim=1)


def save_weights(model: GPT, path: Path) -> None:
    """Write model weights to ``path`` as safetensors (tied weights deduplicated)."""
    _safetensors_save_model(model, str(path))


def load_pretrained(models_dir: Path, device: torch.device) -> tuple[GPT, Tokenizer]:
    """Rebuild the trained model and tokenizer from a ``models/`` artifact directory.

    Reads ``config.json`` to reconstruct the exact architecture, loads the
    safetensors weights onto ``device`` in eval mode, and loads the tokenizer.
    """
    config = GPTConfig.load(models_dir / MODEL_CONFIG_FILENAME)
    model = GPT(config)
    _safetensors_load_model(model, str(models_dir / WEIGHTS_FILENAME))
    model.to(device).eval()
    tokenizer = load_tokenizer(models_dir / TOKENIZER_FILENAME)
    return model, tokenizer
