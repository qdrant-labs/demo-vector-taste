"""Rich per-segment descriptors, scored with CLAP against a curated vocabulary.

Why this exists
---------------
The corpus carries **9 distinct tags across 1,006 segments** (electronic, hip-hop,
experimental, folk, instrumental, rock, pop, international, blues). Prompts built from that
collapse to the same handful of phrases no matter what the user picked — three of the four
originally baked prompts contained "hip hop with a heavy beat, electronic production".

Instead of sourcing more metadata, ask the model we already have. CLAP puts audio and text
in one space, so scoring a track against arbitrary descriptive phrases is just a dot
product. No audio is re-read: this runs against the vectors already stored in Qdrant, which
makes it a payload update (~a minute) rather than a re-ingest (~15).

Scoring is corpus-relative
--------------------------
Raw cosines are not usable directly. CLAP's audio embeddings share a large common component
(the corpus mean vector has norm ~0.96), so some descriptors score high for *everything* and
the per-category scales differ wildly — a metal track scored every `texture` term negatively
while scoring `heavily distorted` at 0.31.

So each descriptor's score is measured **relative to its mean over the corpus**. That picks
what is distinctive about a track rather than what is generically true of all music, and
makes categories comparable. Same centering insight that exposed the degenerate embedder.
"""

from __future__ import annotations

import numpy as np

from .config import COLLECTION, get_client

# Phrased as caption fragments, not bare keywords: CLAP's text tower was trained on
# captions like "hip hop music with a heavy beat", so "lo-fi and tape-saturated" scores
# more meaningfully than "lofi".
VOCAB: dict[str, list[str]] = {
    "mood": [
        "dreamy and hazy", "aggressive and intense", "melancholy and wistful",
        "uplifting and bright", "hypnotic and trance-like", "menacing and dark",
        "playful and quirky", "sombre and solemn", "euphoric and soaring",
        "tense and uneasy", "warm and nostalgic", "cold and clinical",
    ],
    "instrument": [
        "acoustic guitar", "distorted electric guitar", "grand piano",
        "analog synthesizer", "lush strings", "saxophone", "drum machine",
        "live drum kit", "upright bass", "vinyl crackle and tape hiss",
        "hand percussion", "choral voices", "brass section", "electric organ",
        "plucked banjo or mandolin", "flute or woodwind",
    ],
    "production": [
        "lo-fi and tape-saturated", "clean and polished studio production",
        "heavily distorted and fuzzy", "spacious with long reverb",
        "dry and close-miked", "heavily compressed and loud",
        "raw and unmixed", "wide stereo and shimmering",
    ],
    "texture": [
        "sparse and minimal", "dense and layered", "driving and rhythmic",
        "ambient and floating", "syncopated and groovy", "steady and hypnotic loop",
    ],
}

FLAT: list[tuple[str, str]] = [(cat, t) for cat, terms in VOCAB.items() for t in terms]
TERMS: list[str] = [t for _, t in FLAT]

# How many descriptors to keep per category. Two is enough to be evocative without turning
# the prompt into a word salad that ACE-Step ignores.
TOP_PER_CATEGORY = 2


def vocab_matrix() -> np.ndarray:
    """(n_terms, 512) L2-normalized text embeddings for the vocabulary."""
    from .embed import embed_text

    m = embed_text(TERMS)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def score_matrix(audio: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Corpus-relative descriptor scores: (n_segments, n_terms), mean-centered per term."""
    raw = audio @ vocab.T
    return raw - raw.mean(axis=0, keepdims=True)


def top_descriptors(scores: np.ndarray, per_category: int = TOP_PER_CATEGORY) -> list[str]:
    """Best `per_category` terms in each category for one segment's score row.

    Selecting within category rather than globally keeps one strong category (production
    terms score highest in practice) from crowding out mood and instrument entirely.
    """
    out: list[str] = []
    for cat in VOCAB:
        idx = [i for i, (c, _) in enumerate(FLAT) if c == cat]
        best = sorted(idx, key=lambda i: -scores[i])[:per_category]
        out.extend(TERMS[i] for i in best)
    return out


def describe_collection(batch: int = 256, per_category: int = TOP_PER_CATEGORY) -> dict:
    """Score every segment and write `descriptors` onto its points.

    Reads the audio vectors already in Qdrant — no audio decoding, no CLAP audio pass.
    Re-runnable: iterating the vocabulary is cheap by design.
    """
    client = get_client()
    vocab = vocab_matrix()

    # Descriptors are scored per SEGMENT (from its first chunk) but written to ALL of that
    # segment's points. Search returns whichever chunk scored best, so writing only to
    # chunk 0 would leave most retrieved hits with no descriptors at all — which is exactly
    # what happened the first time. `caption` is duplicated across chunks for the same
    # reason; only the `text` vector is chunk-0-only.
    seg_vec: dict[str, np.ndarray] = {}
    seg_ids: dict[str, list] = {}
    offset = None
    while True:
        recs, offset = client.scroll(
            COLLECTION, limit=512, offset=offset, with_vectors=True,
            with_payload=["segment_id", "chunk_index", "is_generated"],
        )
        for r in recs:
            pl = r.payload or {}
            if pl.get("is_generated"):
                continue
            sid = pl.get("segment_id")
            if sid is None:
                continue
            seg_ids.setdefault(sid, []).append(r.id)
            if pl.get("chunk_index") == 0:
                vec = r.vector.get("audio") if isinstance(r.vector, dict) else r.vector
                if vec is not None:
                    seg_vec[sid] = np.asarray(vec, dtype=np.float32)
        if offset is None:
            break

    segments = [s for s in seg_ids if s in seg_vec]
    if not segments:
        return {"segments": 0, "points": 0, "terms": len(TERMS)}

    scores = score_matrix(np.vstack([seg_vec[s] for s in segments]), vocab)

    done = points_written = 0
    for i, sid in enumerate(segments):
        client.set_payload(
            collection_name=COLLECTION,
            payload={"descriptors": top_descriptors(scores[i], per_category)},
            points=seg_ids[sid],
            wait=False,
        )
        done += 1
        points_written += len(seg_ids[sid])
        if done % batch == 0 or done == len(segments):
            print(f"\r  {done}/{len(segments)} segments described", end="", flush=True)
    print()

    return {"segments": done, "points": points_written, "terms": len(TERMS)}


def describe_vectors(vectors: np.ndarray, per_category: int = TOP_PER_CATEGORY) -> list[list[str]]:
    """Descriptors for arbitrary vectors, scored against the CORPUS mean.

    Used for generated audio, which is not in the collection yet. Centring on the corpus
    keeps the numbers comparable with stored descriptors.
    """
    client = get_client()
    vocab = vocab_matrix()

    recs, _ = client.scroll(COLLECTION, limit=1024, with_vectors=True,
                            with_payload=["chunk_index"])
    corpus = np.vstack([
        np.asarray(r.vector["audio"] if isinstance(r.vector, dict) else r.vector,
                   dtype=np.float32)
        for r in recs if r.vector is not None
    ])
    baseline = (corpus @ vocab.T).mean(axis=0)

    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    return [top_descriptors(row, per_category) for row in (vectors @ vocab.T) - baseline]
