"""API contract: health, full generation, validation errors, and SSE streaming.

Uses a tiny model written to a temp dir (pointed at via MINIGPT_MODELS_DIR) so the
real startup/loading path is exercised while staying fast.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from src.config import (
    MODEL_CONFIG_FILENAME,
    TOKENIZER_FILENAME,
    WEIGHTS_FILENAME,
    GPTConfig,
)
from src.model import GPT, save_weights
from src.tokenizer import CharTokenizer

VOCAB = "abcdefghijklmnopqrstuvwxyz"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    config = GPTConfig(
        vocab_size=len(VOCAB), block_size=16, n_layer=2, n_head=2, d_model=16, dropout=0.0
    )
    torch.manual_seed(0)
    model = GPT(config).eval()
    config.save(tmp_path / MODEL_CONFIG_FILENAME)
    save_weights(model, tmp_path / WEIGHTS_FILENAME)
    CharTokenizer.train(VOCAB).save(tmp_path / TOKENIZER_FILENAME)
    monkeypatch.setenv("MINIGPT_MODELS_DIR", str(tmp_path))

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vocab_size"] == len(VOCAB)
    assert body["num_params"] > 0


def test_generate_returns_completion(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json={"prompt": "abc", "max_new_tokens": 10, "temperature": 1.0, "top_k": 5, "top_p": 0.9},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tokens_generated"] == 10
    assert body["text"].startswith("abc")
    assert len(body["completion"]) == 10  # one char per token for the char tokenizer


def test_generate_rejects_empty_prompt(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422  # Pydantic min_length validation


def test_generate_rejects_unknown_characters(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": "XYZ"})  # uppercase not in vocab
    assert response.status_code == 400


def test_generate_stream_emits_sse_tokens(client: TestClient) -> None:
    response = client.post(
        "/generate/stream",
        json={"prompt": "abc", "max_new_tokens": 5, "temperature": 1.0, "top_k": 5, "top_p": 0.9},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text
    assert '"token"' in response.text
    assert '"done"' in response.text
