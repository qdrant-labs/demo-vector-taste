"""User-supplied audio: validate, embed, upsert, purge.

Deliberately mirrors `ingest.py` so an upload and a corpus track are the same KIND of thing
in the collection -- same chunking, same vector, same payload shape. That is what lets an
upload be searched, marked +/-, and recommended against with no changes to retrieval at all.

Three payload fields differ from a corpus track, and each is load-bearing:

  is_upload      the flag the purge, the percentile population and the UI badge key off.
  license        "your upload", never a CC string. The row subtitle renders this field, and
                 printing "CC-BY" over a stranger's file would be a false licensing claim
                 in a repo whose whole argument is that its corpus is legally clean.
  source_url     empty, so ATTRIBUTIONS.md can never pick it up.

Uploads are purged on every server start. They are somebody else's music sitting inside a
public repo's working tree, and the demo should not accumulate it.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from qdrant_client import models

from . import store
from .config import CHUNK_SEC, COLLECTION, ROOT, SAMPLE_RATE, SEGMENT_SEC, get_client

log = logging.getLogger("vector_taste.uploads")

UPLOADS = ROOT / "uploads"

# Namespace distinct from ingest's, so an upload can never collide with a corpus point.
_NS = uuid.UUID("5f3d9a1e-0002-4000-8000-000000000000")

ALLOWED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MAX_BYTES = 30 * 1024 * 1024
# Longer files are TRUNCATED rather than rejected: someone dropping a 6 minute track should
# get an answer, not an error. 5 minutes is ~30 chunk embeddings.
MAX_SECONDS = 300

IS_UPLOAD = models.Filter(
    must=[models.FieldCondition(key="is_upload", match=models.MatchValue(value=True))]
)
# Every query that must see only the fixed corpus uses this. Retrieval deliberately does NOT
# -- uploads are part of the searchable library.
NOT_UPLOAD = models.Filter(
    must_not=[models.FieldCondition(key="is_upload", match=models.MatchValue(value=True))]
)


class UploadError(ValueError):
    """Rejected input. The message is shown to the user, so it must say what to do."""


def validate(filename: str, size: int) -> str:
    """Check name and size BEFORE anything touches disk. Returns the extension to use."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise UploadError(
            f"{ext or 'that file type'} is not audio we can read — "
            f"use {', '.join(sorted(ALLOWED_EXT))}"
        )
    if size > MAX_BYTES:
        raise UploadError(f"file is {size / 1e6:.0f}MB; the limit is {MAX_BYTES // 1024 // 1024}MB")
    if size == 0:
        raise UploadError("that file is empty")
    return ext


def save(data: bytes, filename: str) -> tuple[Path, str]:
    """Write the bytes under a generated name. Returns (path, track_id).

    The user's filename NEVER reaches the filesystem -- it is display text only. That closes
    path traversal ("../../etc/passwd.mp3") without needing to sanitise anything.
    """
    ext = validate(filename, len(data))
    track_id = f"upload-{uuid.uuid4().hex[:12]}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{track_id}{ext}"
    path.write_bytes(data)
    return path, track_id


def ingest(path: Path, track_id: str, display_name: str) -> dict:
    """Chunk, embed and upsert one uploaded file. Returns the clip descriptor for the UI.

    Raises UploadError if the bytes are not decodable audio -- which is the real file-type
    check. An extension allowlist only stops honest mistakes; this stops a .txt renamed .mp3.
    """
    from . import corpus
    from .embed import chunk_audio, embed_chunks, load_audio
    from .ingest import _segments, segment_id

    try:
        wav = load_audio(path)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same answer
        raise UploadError("could not read that as audio — is it a real music file?") from exc
    if not len(wav):
        raise UploadError("that file decoded to no audio at all")

    truncated = len(wav) > MAX_SECONDS * SAMPLE_RATE
    if truncated:
        wav = wav[: MAX_SECONDS * SAMPLE_RATE]

    bpm, key = corpus.analyze(path)
    store.ensure_collection()

    points, n_segments = [], 0
    for seg_idx, seg_wav in _segments(wav, SAMPLE_RATE):
        chunks = chunk_audio(seg_wav)
        if not chunks:
            continue
        n_segments += 1
        sid = segment_id(track_id, seg_idx)
        for c_idx, vec in enumerate(embed_chunks(chunks)):
            points.append(
                store.make_point(
                    str(uuid.uuid5(_NS, f"{track_id}:{seg_idx}:{c_idx}")),
                    vec,
                    {
                        "track_id": track_id,
                        "segment_id": sid,
                        "segment_index": seg_idx,
                        "chunk_index": c_idx,
                        "artist": "You",
                        "title": display_name[:120],
                        "start_sec": seg_idx * SEGMENT_SEC + c_idx * CHUNK_SEC,
                        "end_sec": seg_idx * SEGMENT_SEC + (c_idx + 1) * CHUNK_SEC,
                        "bpm": bpm,
                        "key": key,
                        "tags": ["upload"],
                        "caption": display_name[:120],
                        "license": "your upload",
                        "license_full": "unknown — user supplied",
                        "source_url": "",
                        "is_generated": False,
                        "generation_run_id": None,
                        "is_upload": True,
                        "audio_path": str(path.relative_to(ROOT)),
                    },
                    # No text vector: we have a filename, not a description of the sound.
                    # Embedding the filename would put noise in the text space.
                )
            )

    if not points:
        raise UploadError("that file is too short to embed (needs at least 3 seconds)")

    store.upsert_points(points)
    log.info("ingested upload %s: %d points across %d segments", track_id, len(points), n_segments)
    return {
        "track_id": track_id,
        "segment_id": segment_id(track_id, 0),
        "title": display_name[:120],
        "point_ids": [p.id for p in points],
        "segments": n_segments,
        "points": len(points),
        "bpm": bpm,
        "key": key,
        "seconds": round(len(wav) / SAMPLE_RATE, 1),
        "truncated": truncated,
        "audio_url": f"/audio/{segment_id(track_id, 0)}",
    }


def listing() -> list[dict]:
    """Every upload currently in the collection, newest last, one entry per track."""
    client = get_client()
    records, _ = client.scroll(
        COLLECTION, scroll_filter=IS_UPLOAD, limit=10_000, with_payload=True
    )
    by_track: dict[str, dict] = {}
    for r in records:
        p = r.payload or {}
        tid = p.get("track_id")
        if not tid:
            continue
        clip = by_track.setdefault(
            tid,
            {
                "track_id": tid,
                "segment_id": f"{tid}:0",
                "title": p.get("title", "upload"),
                "bpm": p.get("bpm"),
                "key": p.get("key"),
                "point_ids": [],
                "points": 0,
                "audio_url": f"/audio/{tid}:0",
            },
        )
        clip["point_ids"].append(str(r.id))
        clip["points"] += 1
    return list(by_track.values())


def delete(track_id: str) -> int:
    """Remove one upload: its points and its file. Returns points deleted."""
    client = get_client()
    flt = models.Filter(
        must=[
            models.FieldCondition(key="track_id", match=models.MatchValue(value=track_id)),
            models.FieldCondition(key="is_upload", match=models.MatchValue(value=True)),
        ]
    )
    n = client.count(COLLECTION, count_filter=flt, exact=True).count
    if n:
        client.delete(COLLECTION, points_selector=models.FilterSelector(filter=flt), wait=True)
    for f in UPLOADS.glob(f"{track_id}.*"):
        f.unlink(missing_ok=True)
    return n


def purge() -> int:
    """Drop every upload. Called on server start so each run begins on the fixed corpus."""
    client = get_client()
    try:
        n = client.count(COLLECTION, count_filter=IS_UPLOAD, exact=True).count
    except Exception as exc:  # noqa: BLE001 - no collection yet is not an error here
        log.debug("upload purge skipped: %s", exc)
        return 0
    if n:
        client.delete(
            COLLECTION, points_selector=models.FilterSelector(filter=IS_UPLOAD), wait=True
        )
    if UPLOADS.exists():
        shutil.rmtree(UPLOADS, ignore_errors=True)
    if n:
        log.info("purged %d upload points", n)
    return n
