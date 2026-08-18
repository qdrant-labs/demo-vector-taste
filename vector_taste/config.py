"""Configuration. One env var pair switches between local Qdrant and Qdrant Cloud."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent

# .env.local wins over .env — the usual convention for machine-specific secrets.
# Loaded first because python-dotenv does not overwrite already-set keys.
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")
RAW = ROOT / "raw"
AUDIO = ROOT / "audio"
DATA = ROOT / "data"
BANK = ROOT / "bank"
# Live generations land here, separate from the curated bank so fresh output never
# overwrites or pollutes the pre-baked stage assets.
GENERATED = ROOT / "generated"
MODELS = ROOT / "models"
TIMINGS = ROOT / "timings.jsonl"

COLLECTION = os.getenv("VT_COLLECTION", "music_segments")

# CLAP: 512-d joint text/audio space, Apache-2.0.
# Pinned by revision so a re-clone six months from now produces identical vectors.
#
# `larger_clap_general`, NOT `larger_clap_music`, despite the latter's better published
# GTZAN score. Measured on this corpus (4 genres, zero-shot text->audio classification):
#
#   larger_clap_general    accuracy 0.75   mean pairwise audio cosine 0.31
#   clap-htsat-unfused     accuracy 0.41   mean pairwise audio cosine 0.55
#   larger_clap_music      accuracy 0.25   mean pairwise audio cosine 0.92   <- chance
#
# The music checkpoint is degenerate here: it emits near-identical vectors for completely
# different tracks (two unrelated songs embed at cosine 0.99), so every text query returns
# the same results. Weights load with no missing or mismatched keys, so this is the
# checkpoint's behavior, not a loading bug. Re-run scripts/eval_embedders.py to re-check.
CLAP_MODEL = os.getenv("VT_CLAP_MODEL", "laion/larger_clap_general")
CLAP_REVISION = os.getenv("VT_CLAP_REVISION") or None
EMBED_DIM = 512

# CLAP's audio tower has a hard 10s window (nb_max_samples=480000 @ 48kHz).
# Longer input gets a RANDOM 10s crop, which makes embeddings non-reproducible,
# so we chunk explicitly instead of letting the processor truncate.
SAMPLE_RATE = 48_000
CHUNK_SEC = 10
SEGMENT_SEC = 30

# Default is `local`: every compose generates a NEW track rather than replaying
# pre-baked audio. Set GEN_BACKEND=bank for the stage, where a ~2 minute wait in
# front of an audience is not acceptable and instant playback matters more than
# freshness.
GEN_BACKEND = os.getenv("GEN_BACKEND", "local")


def get_client() -> QdrantClient:
    """Local podman/docker by default; Qdrant Cloud when QDRANT_URL is set.

    timeout=60 because Cloud's REST default is 5s and bulk upserts exceed it.
    prefer_grpc stays off: Cloud gRPC on 6334 needs TLS and buys nothing at this scale.
    """
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )


def is_cloud() -> bool:
    return bool(os.getenv("QDRANT_API_KEY"))


def get_cloud_client() -> QdrantClient | None:
    """Client for the Cloud parity test only.

    Kept on separate env vars (VT_CLOUD_*) so pointing the parity test at Cloud cannot
    accidentally repoint the whole demo at it and break the offline guarantee.
    Returns None when unconfigured, so the test SKIPs rather than fails.
    """
    url, key = os.getenv("VT_CLOUD_URL"), os.getenv("VT_CLOUD_API_KEY")
    if not (url and key):
        return None
    return QdrantClient(url=url, api_key=key, timeout=60)
