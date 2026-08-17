"""Taste refinement: positives, negatives, and the diff that makes it visible.

This is the centerpiece. The claim being demonstrated is that a human can stake out a
region of embedding space by ear — "more like this, less like that" — and that the result
set visibly moves when they do.

One vector per gesture
----------------------
A 30s segment is stored as three 10s chunk points. When someone marks a segment, we
contribute exactly ONE vector: the chunk that actually surfaced (`Hit.point_id`). Passing
all three would give that segment triple the weight of a single-vector example in
`average_vector` and silently distort the very thing we're demonstrating. One click, one
vote — and it's the honest reading of the gesture, since the user reacted to the part they
heard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import models

from .config import COLLECTION, DATA, get_client
from .embed import centroid as _centroid
from .search import EXACT, NOT_GENERATED, OVERFETCH, Hit, _to_hits, fetch_vectors, merge_filters

# best_score: a group scores as max(similarity to any positive), with negatives squared and
# negated when they win. Chosen over average_vector because a single added negative visibly
# reorders the list instead of nudging a mean — which is the whole point on stage.
# average_vector collapses several positives into one query point, so distinct tastes
# average into something matching neither.
DEFAULT_STRATEGY = models.RecommendStrategy.BEST_SCORE


@dataclass
class TasteProfile:
    positives: list[str] = field(default_factory=list)  # point IDs
    negatives: list[str] = field(default_factory=list)
    steer: str = ""

    @property
    def hash(self) -> str:
        """Stable ID for the bank lookup. Order-independent: the same set of gestures in a
        different order is the same taste."""
        payload = json.dumps(
            {
                "pos": sorted(self.positives),
                "neg": sorted(self.negatives),
                "steer": self.steer.strip().lower(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def is_empty(self) -> bool:
        return not self.positives and not self.negatives

    def to_dict(self) -> dict:
        return {
            "positives": self.positives,
            "negatives": self.negatives,
            "steer": self.steer,
            "hash": self.hash,
        }

    def save(self, name: str | None = None):
        DATA.mkdir(parents=True, exist_ok=True)
        path = DATA / f"taste_{name or self.hash}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, name: str) -> TasteProfile:
        path = DATA / f"taste_{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"no taste profile at {path}")
        d = json.loads(path.read_text())
        return cls(d.get("positives", []), d.get("negatives", []), d.get("steer", ""))


def recommend(
    profile: TasteProfile,
    limit: int = 10,
    strategy: models.RecommendStrategy = DEFAULT_STRATEGY,
    flt: models.Filter | None = None,
) -> list[Hit]:
    """Recommendation API with multiple positives AND negatives, grouped to segments."""
    if profile.is_empty():
        raise ValueError("taste profile has no examples")

    res = get_client().query_points_groups(
        collection_name=COLLECTION,
        query=models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=list(profile.positives),
                negative=list(profile.negatives),
                strategy=strategy,
            )
        ),
        using="audio",
        group_by="segment_id",
        limit=limit * OVERFETCH,
        group_size=3,
        query_filter=merge_filters(flt),
        search_params=EXACT,
        with_payload=True,
    )
    return _to_hits(res.groups)[:limit]


def discover(target: str, context: list[tuple[str, str]], limit: int = 10) -> list[Hit]:
    """Discovery API: a target plus (positive, negative) context pairs.

    Different intent from recommend(): each pair partitions the space rather than pulling
    toward an average. Use it for "in the direction of A rather than B, near target".
    """
    res = get_client().query_points_groups(
        collection_name=COLLECTION,
        query=models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=target,
                context=[models.ContextPair(positive=p, negative=n) for p, n in context],
            )
        ),
        using="audio",
        group_by="segment_id",
        limit=limit * OVERFETCH,
        group_size=3,
        query_filter=NOT_GENERATED,
        search_params=EXACT,
        with_payload=True,
    )
    return _to_hits(res.groups)[:limit]


def negative_hits(profile: TasteProfile) -> list[Hit]:
    """The negative examples as Hits, so prompt synthesis can contrast against them.

    Retrieved rather than re-queried: negatives are point IDs the user already marked, and
    all we need is their payload (descriptors, tags) to subtract from the positive side.
    """
    if not profile.negatives:
        return []
    recs = get_client().retrieve(COLLECTION, ids=profile.negatives, with_payload=True)
    return [
        Hit(
            segment_id=(r.payload or {}).get("segment_id", str(r.id)),
            score=0.0,
            point_id=str(r.id),
            payload=r.payload or {},
            start_sec=int((r.payload or {}).get("start_sec", 0)),
            n_chunks=1,
        )
        for r in recs
    ]


def taste_centroid(profile: TasteProfile) -> np.ndarray:
    """Mean of the positive chunk vectors, re-normalized.

    Negatives intentionally do NOT move the centroid. They shape *retrieval* (via
    best_score), but the centroid is "where the user pointed", and subtracting negatives
    from it would push it into empty space that no real music occupies — making the final
    distance number meaningless.
    """
    if not profile.positives:
        raise ValueError("cannot build a centroid without positive examples")
    vecs = fetch_vectors(profile.positives)
    missing = [p for p in profile.positives if p not in vecs]
    if missing:
        raise ValueError(f"positives not found in collection: {missing}")
    return _centroid(np.vstack([vecs[p] for p in profile.positives]))


@dataclass
class Diff:
    """What changed between two result sets."""

    dropped: list[Hit]
    added: list[Hit]
    moved: list[tuple[Hit, int, int]]  # hit, old rank, new rank
    kept: int

    @property
    def changed(self) -> bool:
        return bool(self.dropped or self.added or self.moved)


def diff(before: list[Hit], after: list[Hit]) -> Diff:
    """Compare two result sets by segment_id."""
    b_rank = {h.segment_id: i for i, h in enumerate(before)}
    a_rank = {h.segment_id: i for i, h in enumerate(after)}
    a_by_id = {h.segment_id: h for h in after}

    dropped = [h for h in before if h.segment_id not in a_rank]
    added = [h for h in after if h.segment_id not in b_rank]
    moved = [
        (a_by_id[sid], b_rank[sid], a_rank[sid])
        for sid in a_rank
        if sid in b_rank and b_rank[sid] != a_rank[sid]
    ]
    moved.sort(key=lambda t: abs(t[1] - t[2]), reverse=True)
    kept = len(set(b_rank) & set(a_rank)) - len(moved)
    return Diff(dropped=dropped, added=added, moved=moved, kept=kept)


def format_diff(d: Diff) -> str:
    """Text diff for the CLI. The UI renders the same data.

    Every state carries a text marker, never colour alone — this gets shown on projectors.
    """
    lines = []
    if not d.changed:
        lines.append("  result set did NOT change")
        return "\n".join(lines)

    for h in d.dropped:
        lines.append(f"  [-] DROPPED           {h.label[:50]}")
    for h in d.added:
        lines.append(f"  [+] NEW      {h.score:>7.4f}  {h.label[:50]}")
    for h, old, new in d.moved:
        arrow = "up" if new < old else "down"
        lines.append(
            f"  [~] {arrow:<4} #{old + 1}->#{new + 1}  {h.score:>7.4f}  {h.label[:44]}"
        )
    lines.append(
        f"  ({len(d.added)} in, {len(d.dropped)} out, {len(d.moved)} moved, {d.kept} unchanged)"
    )
    return "\n".join(lines)


def percentile_against_centroid(
    vector: np.ndarray, centroid_vec: np.ndarray
) -> tuple[float, float, int]:
    """Score a vector against the taste centroid, as a percentile of the human corpus.

    Returns (cosine, percentile, population_size).

    Three choices make this number mean something rather than being unfalsifiable:
      - Population is SEGMENTS, not chunks: each segment scores as its best chunk, the same
        max-sim rule retrieval uses. Ranking a generated track's one good chunk against
        other tracks' filler would inflate it.
      - Population EXCLUDES generated points, so the comparison is against human music and
        does not drift as generated tracks accumulate across rehearsal runs.
      - The subject is scored the same way, so it is compared like with like.
    """
    client = get_client()
    total = client.count(COLLECTION, count_filter=NOT_GENERATED, exact=True).count
    res = client.query_points(
        collection_name=COLLECTION,
        query=centroid_vec.tolist(),
        using="audio",
        limit=max(total, 1),
        query_filter=NOT_GENERATED,
        search_params=EXACT,
        with_payload=["segment_id"],
        with_vectors=False,
    )
    best_per_segment: dict[str, float] = {}
    for p in res.points:
        sid = (p.payload or {}).get("segment_id", str(p.id))
        if p.score > best_per_segment.get(sid, -2.0):
            best_per_segment[sid] = float(p.score)

    scores = np.array(list(best_per_segment.values()), dtype=np.float32)
    cos = float(np.dot(vector, centroid_vec))
    pct = float((scores < cos).mean() * 100) if len(scores) else 0.0
    return cos, pct, len(scores)
