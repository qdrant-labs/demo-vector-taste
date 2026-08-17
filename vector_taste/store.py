"""Qdrant collection: named vectors, payload indexes, upsert.

Two API facts worth stating, because most published examples are wrong:

- `client.search()`, `.recommend()`, and `.discover()` were REMOVED in qdrant-client 1.16.0
  (not deprecated). Everything goes through `query_points()` / `query_points_groups()`.
- Qdrant Cloud enables strict mode by default with `unindexed_filtering_retrieve=false`,
  so filtering or grouping on an un-indexed payload field works locally and ERRORS on
  Cloud. Every field we filter or group on is indexed in `INDEXES` below. That single list
  is what makes this demo Cloud-safe.
"""

from __future__ import annotations

import numpy as np
from qdrant_client import models

from .config import COLLECTION, EMBED_DIM, get_client

# Every field here is either filtered or grouped on somewhere in the app.
# Adding a filter without adding its index here will pass locally and fail on Cloud.
INDEXES: dict[str, models.PayloadSchemaType] = {
    "segment_id": models.PayloadSchemaType.KEYWORD,  # group_by for chunk -> segment rollup
    "track_id": models.PayloadSchemaType.KEYWORD,
    "artist": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "bpm": models.PayloadSchemaType.INTEGER,
    "is_generated": models.PayloadSchemaType.BOOL,  # excludes generated from retrieval
}

# Named vectors. `lyrics` is deliberately absent in v1: it would be a third named vector
# fed by a lyrics-aware encoder (CLAP's text tower is trained on captions, not lyrics),
# and it would slot in right here with its own dimension.
VECTORS = {
    "audio": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
    "text": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
}


def ensure_collection(recreate: bool = False) -> None:
    client = get_client()
    exists = client.collection_exists(COLLECTION)

    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VECTORS,
            # One shard keeps result ordering stable: cross-shard merges are another
            # place equal-scored points can reorder between runs.
            shard_number=1,
        )

    # Idempotent: creating an existing index is a no-op we can safely ignore.
    for field, schema in INDEXES.items():
        try:
            client.create_payload_index(COLLECTION, field, field_schema=schema)
        except Exception as exc:  # noqa: BLE001
            if "already exists" not in str(exc).lower():
                raise


def upsert_points(points: list[models.PointStruct], batch: int = 128) -> int:
    """Batched upsert. Cloud's REST timeout makes one giant request a bad idea."""
    client = get_client()
    for i in range(0, len(points), batch):
        client.upsert(collection_name=COLLECTION, points=points[i : i + batch], wait=True)
    return len(points)


def make_point(
    point_id: str,
    audio_vec: np.ndarray,
    payload: dict,
    text_vec: np.ndarray | None = None,
) -> models.PointStruct:
    """One point per 10s chunk.

    The `text` vector is a SEGMENT-level property, so it is attached only to the segment's
    first chunk. Putting it on all three would make a text query return the same segment
    three times with identical scores.
    """
    vector: dict[str, list[float]] = {"audio": [float(x) for x in audio_vec]}
    if text_vec is not None:
        vector["text"] = [float(x) for x in text_vec]
    return models.PointStruct(id=point_id, vector=vector, payload=payload)


def count(only_generated: bool | None = None) -> int:
    flt = None
    if only_generated is not None:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="is_generated", match=models.MatchValue(value=only_generated)
                )
            ]
        )
    return get_client().count(COLLECTION, count_filter=flt, exact=True).count


def delete_generated() -> int:
    """Purge generated points.

    Needed because generated points would otherwise become search neighbours and shift the
    finale percentile on the second rehearsal run - a bug that only appears on a re-run.
    """
    n = count(only_generated=True)
    if n:
        get_client().delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="is_generated", match=models.MatchValue(value=True)
                        )
                    ]
                )
            ),
            wait=True,
        )
    return n


def collection_info() -> dict:
    info = get_client().get_collection(COLLECTION)
    return {
        "points": info.points_count,
        # Expect 0 at demo scale: HNSW only builds past ~5000 vectors per segment, so
        # search is exact brute force. That is desirable here - it is deterministic.
        "indexed_vectors": info.indexed_vectors_count,
        "status": str(info.status),
    }
