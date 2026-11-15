# The web app. Corpus audio, Qdrant data and the CLAP weights are mounted, never baked in —
# see docker-compose.yml. Search and the ElevenLabs backend work here; GEN_BACKEND=local
# (ACE-Step on MLX) needs the host's GPU and does not run in a Linux container.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 VIRTUAL_ENV=/app/.venv PATH="/app/.venv/bin:$PATH"

# Locked versions, but torch from the CPU index: PyPI's Linux torch drags in ~6 GB of
# NVIDIA CUDA libraries that a container on a Mac can never use.
COPY pyproject.toml uv.lock README.md ./
RUN uv venv \
    && uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/all.txt \
    && grep -vE '^(torch|torchaudio|nvidia-|triton|cuda-)' /tmp/all.txt > /tmp/req.txt \
    && uv pip install -r /tmp/req.txt \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu \
         "torch==$(grep -E '^torch==' /tmp/all.txt | cut -d= -f3 | cut -d' ' -f1)" \
         "torchaudio==$(grep -E '^torchaudio==' /tmp/all.txt | cut -d= -f3 | cut -d' ' -f1)"

# Editable on purpose: config.ROOT is the package's parent, which must be /app for the
# mounted audio/, data/ and generated/ to be found.
COPY vector_taste ./vector_taste
RUN uv pip install --no-deps -e .

EXPOSE 8000
CMD ["vt", "ui", "--host", "0.0.0.0", "--port", "8000"]
