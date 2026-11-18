# Multi-stage Dockerfile for tts_gateway.
#
# w2v-common resolves from the public git source declared in
# pyproject.toml. No browser/ML deps — this is a thin queue + API service; the
# heavy kokoro synthesis lives in the remote workers, never here.

ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl ca-certificates libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
COPY worker/ ./worker/
COPY alembic.ini ./alembic.ini
RUN uv sync --frozen --no-dev

FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 ttsgateway && mkdir -p /app && chown -R ttsgateway:ttsgateway /app
WORKDIR /app
COPY --from=builder --chown=ttsgateway:ttsgateway /app/.venv /app/.venv
COPY --chown=ttsgateway:ttsgateway src/ ./src/
COPY --chown=ttsgateway:ttsgateway worker/ ./worker/
COPY --chown=ttsgateway:ttsgateway alembic.ini ./alembic.ini
ENV PATH="/app/.venv/bin:$PATH"
USER 1000:1000

# Port 6320. k8s probes are the source of truth for health (no Dockerfile
# HEALTHCHECK). Alembic runs in a k8s initContainer (alembic: true), not the CMD.
EXPOSE 6320
CMD ["python", "-m", "src.asgi"]
