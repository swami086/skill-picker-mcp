# syntax=docker/dockerfile:1

# Multi-stage: build venv → slim runtime (docker-development pattern).
# stdio MCP: Cursor/Claude spawn `docker run -i --rm … skill-picker-mcp`.

FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# CPU torch via [tool.uv.sources] pytorch-cpu index (see pyproject.toml).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
 && uv sync --frozen --no-dev --no-editable \
 && /opt/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Bake MiniLM so first query does not hit the network.
ENV HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf
RUN /opt/venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('model ok')"

# ── runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -r -g 1000 app \
 && useradd -r -u 1000 -g app -d /home/app -m app \
 && mkdir -p /skills /data /opt/hf \
 && chown -R app:app /data /home/app /opt/hf

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf /opt/hf

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf \
    SKILL_PICKER_SKILLS_ROOT=/skills \
    SKILL_PICKER_CATALOG=/skills/skills-catalog.json \
    SKILL_PICKER_DB=/data/skills.db \
    SKILL_PICKER_HOST_PREFIX=~/.cursor/skills \
    HOME=/home/app

WORKDIR /home/app
USER app

VOLUME ["/skills", "/data"]
# stdio only — no EXPOSE
ENTRYPOINT ["skill-picker-mcp"]
