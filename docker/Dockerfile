# One parametrised image for all Python services: compose passes REQUIREMENTS and the start command.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS=requirements/common.txt
ARG TORCH_CPU=false
ARG SPACY_MODEL=""
ARG ST_MODEL=""
COPY requirements/ requirements/
COPY requirements-dev.txt ./
# CPU-only torch keeps the RAG image ~2 GB smaller than the default CUDA wheel.
RUN if [ "$TORCH_CPU" = "true" ]; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi \
    && pip install -r "$REQUIREMENTS"
RUN if [ -n "$SPACY_MODEL" ]; then python -m spacy download "$SPACY_MODEL"; fi
RUN if [ -n "$ST_MODEL" ]; then python -c "from sentence_transformers import SentenceTransformer as S; S('$ST_MODEL')"; fi

COPY src/ src/
COPY configs/ configs/
COPY data/policies/ data/policies/
COPY notebooks/ notebooks/
COPY scripts/ scripts/
RUN useradd -m -u 10001 app && mkdir -p /app/artifacts /app/data/raw && chown -R app /app
USER app
