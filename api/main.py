"""FastAPI service: load the model once, then serve full and streaming generation.

Handlers stay thin: request validation is done by Pydantic and all generation
logic is delegated to ``src``. The demo UI is served as static files from the
same app so a single deployment provides both the API and a live page.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, PROJECT_ROOT, GenerationConfig, resolve_device
from src.generate import iter_generate
from src.model import load_pretrained
from src.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_DEFAULTS = GenerationConfig()
_DEMO_DIR = PROJECT_ROOT / "demo"


class GenerateRequest(BaseModel):
    """Validated generation parameters; bounds keep a public demo well-behaved."""

    prompt: str = Field(min_length=1, description="Text prompt to continue.")
    max_new_tokens: int = Field(default=_DEFAULTS.max_new_tokens, ge=1, le=1024)
    temperature: float = Field(default=_DEFAULTS.temperature, ge=0.0, le=2.0)
    top_k: int | None = Field(default=_DEFAULTS.top_k, ge=0)
    top_p: float | None = Field(default=_DEFAULTS.top_p, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    text: str
    tokens_generated: int
    latency_ms: float
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    vocab_size: int
    num_params: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and tokenizer exactly once when the server starts."""
    models_dir = Path(os.environ.get("MINIGPT_MODELS_DIR", str(MODELS_DIR)))
    device = resolve_device()
    logger.info("loading model from %s on %s", models_dir, device)
    model, tokenizer = load_pretrained(models_dir, device)
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.device = device
    logger.info("model ready: %.2fM params", model.num_parameters() / 1e6)
    yield


app = FastAPI(title="MiniGPT", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_prompt(tokenizer: Tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    try:
        ids = tokenizer.encode(prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not ids:
        raise HTTPException(status_code=400, detail="prompt is empty after tokenization")
    return torch.tensor([ids], dtype=torch.long, device=device)


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    return HealthResponse(
        status="ok",
        device=str(request.app.state.device),
        vocab_size=model.config.vocab_size,
        num_params=model.num_parameters(),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device

    prompt_ids = _encode_prompt(tokenizer, body.prompt, device)
    start = time.perf_counter()
    output = model.generate(
        prompt_ids, body.max_new_tokens, body.temperature, body.top_k, body.top_p
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    text = tokenizer.decode(output[0].tolist())
    tokens_generated = output.shape[1] - prompt_ids.shape[1]
    return GenerateResponse(
        prompt=body.prompt,
        completion=text[len(body.prompt) :],
        text=text,
        tokens_generated=tokens_generated,
        latency_ms=latency_ms,
        device=str(device),
    )


@app.post("/generate/stream")
def generate_stream(request: Request, body: GenerateRequest) -> StreamingResponse:
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device
    prompt_ids = _encode_prompt(tokenizer, body.prompt, device)

    def event_stream() -> Iterator[str]:
        generated: list[int] = []
        emitted = ""
        for next_id in iter_generate(
            model, prompt_ids, body.max_new_tokens, body.temperature, body.top_k, body.top_p
        ):
            generated.append(int(next_id))
            # Decode the whole suffix each step so multi-byte tokens emit only once
            # a full character is formed.
            text = tokenizer.decode(generated)
            delta = text[len(emitted) :]
            emitted = text
            if delta:
                yield _sse({"token": delta})
        yield _sse({"done": True, "tokens_generated": len(generated)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Mounted last so the explicit API routes above take precedence.
app.mount("/", StaticFiles(directory=str(_DEMO_DIR), html=True), name="demo")
