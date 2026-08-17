"""CLAP embedding: audio and text into one 512-d space.

Two non-obvious things drive this module.

1. CLAP's audio tower has a hard 10-second window. Feeding it 30s gets you a *random*
   10s crop (`truncation: rand_trunc`), so the same file embeds differently every run.
   We chunk into explicit 10s windows and store one point per chunk. Rolling chunks back
   up to segments happens at query time with `query_points_groups`, which scores a group
   by its best member — max-similarity, for free, from the database.

2. `transformers` v5 changed the API. `get_audio_features()` returns a
   `BaseModelOutputWithPooling` (vector in `.pooler_output`) rather than a tensor, and the
   v4 `audios=` kwarg is now `audio=`. The published HF docs still show the v4 form.
   `_vec()` handles both so this keeps working across the boundary.

Outputs are already L2-normalized inside both getters, so we never re-normalize a single
embedding. Cosine == dot product.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from .config import CHUNK_SEC, CLAP_MODEL, CLAP_REVISION, EMBED_DIM, SAMPLE_RATE

# Chunks shorter than this are dropped: a 0.5s tail is mostly silence and its embedding
# is noise that would pollute a segment's max-similarity score.
MIN_CHUNK_SEC = 3
_MIN_SAMPLES = MIN_CHUNK_SEC * SAMPLE_RATE
_CHUNK_SAMPLES = CHUNK_SEC * SAMPLE_RATE


def _vec(out):
    """transformers v5 returns an output object; v4 returned the tensor directly."""
    return getattr(out, "pooler_output", out)


@functools.lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoProcessor, ClapModel

    kwargs = {"revision": CLAP_REVISION} if CLAP_REVISION else {}
    model = ClapModel.from_pretrained(CLAP_MODEL, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(CLAP_MODEL, **kwargs)

    # MPS gives a solid speedup on Apple Silicon and the model is only ~776MB, so it
    # always fits. CPU fallback keeps this working on any machine.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device


def device() -> str:
    return _load()[2]


def chunk_audio(wav: np.ndarray) -> list[np.ndarray]:
    """Split mono audio into CLAP-sized windows, dropping a too-short tail."""
    chunks = [wav[i : i + _CHUNK_SAMPLES] for i in range(0, len(wav), _CHUNK_SAMPLES)]
    kept = [c for c in chunks if len(c) >= _MIN_SAMPLES]
    # A file shorter than MIN_CHUNK_SEC would otherwise embed to nothing at all.
    if not kept and len(wav):
        kept = [wav]
    return kept


def load_audio(path) -> np.ndarray:
    """Mono float32 at CLAP's expected 48kHz."""
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def embed_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Embed pre-chunked audio -> (n_chunks, 512), each row L2-normalized."""
    import torch

    if not chunks:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    model, processor, dev = _load()
    inputs = processor(audio=chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = _vec(model.get_audio_features(**inputs))
    return feats.float().cpu().numpy()


def embed_audio_file(path) -> np.ndarray:
    """One file -> (n_chunks, 512). Deterministic: no random cropping anywhere."""
    return embed_chunks(chunk_audio(load_audio(path)))


def embed_text(texts: str | list[str]) -> np.ndarray:
    """Text -> (n, 512) in the SAME space as audio, so cosine across modalities is valid."""
    import torch

    if isinstance(texts, str):
        texts = [texts]
    model, processor, dev = _load()
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        feats = _vec(model.get_text_features(**inputs))
    return feats.float().cpu().numpy()


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Mean of unit vectors, re-normalized — the taste centroid.

    Re-normalizing matters: the mean of unit vectors is not itself a unit vector, and an
    un-normalized centroid makes cosine scores incomparable between taste profiles built
    from different numbers of examples.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError("centroid of opposing vectors is zero; cannot normalize")
    return (mean / norm).astype(np.float32)
