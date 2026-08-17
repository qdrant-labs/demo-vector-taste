"""Search: text->audio, audio->audio, and both combined.

Everything goes through `query_points_groups(group_by="segment_id")`. Two reasons:

1. It rolls 10s chunk points back up to 30s segments, which is the unit a human listens to.
2. A group is scored by its BEST member, so we get max-similarity aggregation from the
   database. Mean-pooling a segment's chunks would bury the hook: a segment whose chorus
   scores 0.8 and whose intro scores 0.0 reports ~0.46 pooled, but 0.8 under max-sim.

`exact=True` is passed everywhere. At this corpus size Qdrant builds no HNSW graph anyway
(threshold is ~5000 vectors per segment), so this costs nothing and states the intent — and
it keeps results identical if the corpus later grows past that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qdrant_client import models

from .config import COLLECTION, get_client
from .embed import embed_audio_file, embed_text

EXACT = models.SearchParams(exact=True)

# Excludes generated tracks from retrieval. Without this, generated points become search
# neighbours on the second run and quietly shift every score.
NOT_GENERATED = models.Filter(
    must_not=[
        models.FieldCondition(key="is_generated", match=models.MatchValue(value=True))
    ]
)


@dataclass
class Hit:
    """One segment: the group, scored by its best chunk."""

    segment_id: str
    score: float
    point_id: str  # the winning chunk — this is what a +/- gesture contributes
    payload: dict
    start_sec: int
    n_chunks: int

    @property
    def label(self) -> str:
        p = self.payload
        return f"{p.get('artist', '?')} — {p.get('title', '?')}"


def _to_hits(groups) -> list[Hit]:
    """Group -> Hit, with a deterministic tie-break.

    Qdrant's ScoredPoint ordering compares score only, with no point-id tie-break, so
    equal-scored results can reorder between restarts depending on segment layout. Sorting
    here makes rehearsal runs byte-identical.
    """
    hits = []
    for g in groups:
        if not g.hits:
            continue
        best = g.hits[0]
        hits.append(
            Hit(
                segment_id=str(g.id),
                score=float(best.score),
                point_id=str(best.id),
                payload=best.payload or {},
                start_sec=int((best.payload or {}).get("start_sec", 0)),
                n_chunks=len(g.hits),
            )
        )
    return sorted(hits, key=lambda h: (-h.score, h.segment_id))


def _query(query, using: str, limit: int, extra_filter: models.Filter | None = None):
    flt = NOT_GENERATED
    if extra_filter is not None:
        flt = models.Filter(
            must=(extra_filter.must or []) + (NOT_GENERATED.must or []),
            must_not=(extra_filter.must_not or []) + (NOT_GENERATED.must_not or []),
            should=extra_filter.should,
        )
    return get_client().query_points_groups(
        collection_name=COLLECTION,
        query=query,
        using=using,
        group_by="segment_id",
        limit=limit,
        group_size=3,  # a 30s segment holds at most three 10s chunks
        query_filter=flt,
        search_params=EXACT,
        with_payload=True,
    )


def by_text(text: str, limit: int = 10, flt: models.Filter | None = None) -> list[Hit]:
    """Text query against the AUDIO vectors — the cross-modal search that makes the point.

    Deliberately searches `using="audio"`, not `using="text"`: CLAP puts both towers in one
    space, so a text embedding can retrieve audio directly. Searching the text vectors
    would only compare the query to our own captions, which is ordinary keyword search
    wearing a costume.
    """
    return _to_hits(_query(embed_text(text)[0].tolist(), "audio", limit, flt).groups)


def by_audio(path, limit: int = 10, flt: models.Filter | None = None) -> list[Hit]:
    """Audio query. Uses the clip's strongest chunk rather than an average of all of them."""
    vecs = embed_audio_file(path)
    if not len(vecs):
        return []
    return _to_hits(_query(vecs[0].tolist(), "audio", limit, flt).groups)


def combined(
    text: str, path, limit: int = 10, text_weight: float = 0.5, flt: models.Filter | None = None
) -> list[Hit]:
    """Text + audio in one query, as a weighted sum in the shared space.

    Chosen over Qdrant's prefetch/fusion (RRF) after comparing both: RRF blends *rankings*
    and throws away the magnitudes, which is what actually carries "how much like this" in
    a joint embedding space. Summing the unit vectors and re-normalizing keeps the geometry
    and gives a single interpretable knob. Frozen deliberately — not a live toggle.
    """
    t = embed_text(text)[0]
    a = embed_audio_file(path)
    if not len(a):
        return by_text(text, limit, flt)
    mix = text_weight * t + (1.0 - text_weight) * a[0]
    norm = np.linalg.norm(mix)
    if norm == 0:
        return by_text(text, limit, flt)
    return _to_hits(_query((mix / norm).tolist(), "audio", limit, flt).groups)


def fetch_vectors(point_ids: list[str]) -> dict[str, np.ndarray]:
    """Retrieve stored audio vectors by point ID — needed to build a taste centroid."""
    recs = get_client().retrieve(COLLECTION, ids=point_ids, with_vectors=True)
    out = {}
    for r in recs:
        vec = r.vector.get("audio") if isinstance(r.vector, dict) else r.vector
        if vec is not None:
            out[str(r.id)] = np.asarray(vec, dtype=np.float32)
    return out


def format_table(hits: list[Hit], title: str = "") -> str:
    if not hits:
        return f"{title}\n  (no results)" if title else "  (no results)"
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  {'#':>2}  {'score':>7}  {'@s':>4}  {'bpm':>4}  artist — title")
    lines.append("  " + "-" * 74)
    for i, h in enumerate(hits, 1):
        bpm = h.payload.get("bpm")
        lines.append(
            f"  {i:>2}  {h.score:>7.4f}  {h.start_sec:>4}  "
            f"{(bpm if bpm else '-'):>4}  {h.label[:52]}"
        )
    return "\n".join(lines)
