# Minimal CPU image for serving the MiniGPT API + demo.
FROM python:3.13-slim

WORKDIR /app

# Install the CPU-only torch wheel first (much smaller than the default CUDA
# build), then the rest of the pinned runtime stack.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Application code, demo UI, and trained artifacts.
COPY src ./src
COPY api ./api
COPY demo ./demo
COPY models ./models

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app \
    MINIGPT_MODELS_DIR=/app/models \
    PORT=8000

EXPOSE 8000

# Honor the platform-provided $PORT (Render/Fly/Spaces) and default to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
