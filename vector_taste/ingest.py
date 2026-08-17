"""Ingest: audio -> 10s chunks -> CLAP -> Qdrant, plus ATTRIBUTIONS.md.

Point IDs are deterministic UUID5s derived from (track_id, segment, chunk). That matters
more than it looks: re-running ingest updates the same points instead of duplicating them,
and a saved taste profile referencing a point ID stays valid across a rebuild.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

from . import corpus, store
from .config import AUDIO, CHUNK_SEC, DATA, ROOT, SEGMENT_SEC
from .embed import embed_chunks, load_audio
from .timing import stage

_NS = uuid.UUID("5f3d9a1e-0000-4000-8000-000000000000")


def point_id(track_id: str, segment_index: int, chunk_index: int) -> str:
    return str(uuid.uuid5(_NS, f"{track_id}:{segment_index}:{chunk_index}"))


def segment_id(track_id: str, segment_index: int) -> str:
    return f"{track_id}:{segment_index}"


def _segments(wav: np.ndarray, sr: int) -> list[tuple[int, np.ndarray]]:
    """Split into SEGMENT_SEC windows. FMA clips are already 30s, so this is usually one."""
    n = SEGMENT_SEC * sr
    out = [(i // n, wav[i : i + n]) for i in range(0, len(wav), n)]
    # Drop a trailing stub shorter than one chunk; it cannot produce a usable embedding.
    return [(i, s) for i, s in out if len(s) >= CHUNK_SEC * sr] or out[:1]


def ingest_tracks(
    rows: list[dict], analyze_audio: bool = True, flush_every: int = 50
) -> dict:
    """Embed and upsert. Returns counts plus the rows actually ingested.

    Upserts every `flush_every` tracks rather than once at the end: a thousand-track run is
    long enough that an interruption should not throw away all the work, and point IDs are
    deterministic so resuming re-writes the same points instead of duplicating them.
    """
    from .config import SAMPLE_RATE
    from .embed import chunk_audio

    store.ensure_collection()

    points, ingested = [], []
    skipped_missing = 0
    total_points = 0
    vectors_for_disk: list[np.ndarray] = []
    t0 = time.perf_counter()

    def flush():
        nonlocal points, total_points
        if points:
            store.upsert_points(points)
            total_points += len(points)
            points = []

    for n, row in enumerate(rows, 1):
        path = corpus.fma_audio_path(row["track_id"])
        if not path.exists():
            skipped_missing += 1
            continue

        try:
            wav = load_audio(path)
        except Exception:
            skipped_missing += 1
            continue
        if not len(wav):
            skipped_missing += 1
            continue

        bpm, key = corpus.analyze(path) if analyze_audio else (None, None)
        row = {**row, "bpm": bpm, "key": key}
        track = corpus.Track(
            track_id=row["track_id"],
            artist=row["artist"],
            title=row["title"],
            path=path,
            license=row["license"],
            source_url=row["source_url"],
            tags=row["tags"],
            bpm=bpm,
            key=key,
        )
        caption = track.caption

        from .embed import embed_text

        text_vec = embed_text(caption)[0]

        for seg_idx, seg_wav in _segments(wav, SAMPLE_RATE):
            chunks = chunk_audio(seg_wav)
            if not chunks:
                continue
            vecs = embed_chunks(chunks)
            sid = segment_id(track.track_id, seg_idx)
            for c_idx, vec in enumerate(vecs):
                payload = {
                    "track_id": track.track_id,
                    "segment_id": sid,
                    "segment_index": seg_idx,
                    "chunk_index": c_idx,
                    "artist": track.artist,
                    "title": track.title,
                    "start_sec": seg_idx * SEGMENT_SEC + c_idx * CHUNK_SEC,
                    "end_sec": seg_idx * SEGMENT_SEC + (c_idx + 1) * CHUNK_SEC,
                    "bpm": bpm,
                    "key": key,
                    "tags": track.tags,
                    "caption": caption,
                    "license": row["license_short"],
                    "license_full": track.license,
                    "source_url": track.source_url,
                    "is_generated": False,
                    "generation_run_id": None,
                    "audio_path": str(path.relative_to(ROOT)),
                }
                points.append(
                    store.make_point(
                        point_id(track.track_id, seg_idx, c_idx),
                        vec,
                        payload,
                        # text vector only on the segment's first chunk, else a text query
                        # returns the same segment three times
                        text_vec=text_vec if c_idx == 0 else None,
                    )
                )
                vectors_for_disk.append(vec)
        ingested.append(row)

        if n % flush_every == 0:
            flush()
            rate = n / max(time.perf_counter() - t0, 1e-6)
            eta = (len(rows) - n) / rate if rate else 0
            print(
                f"\r  {n}/{len(rows)} tracks  {total_points} points  "
                f"{rate:.1f}/s  eta {eta / 60:.1f}m",
                end="",
                flush=True,
            )

    flush()
    print()

    if vectors_for_disk:
        DATA.mkdir(parents=True, exist_ok=True)
        np.save(DATA / "embeddings.npy", np.vstack(vectors_for_disk).astype(np.float32))

    return {
        "tracks": len(ingested),
        "points": total_points,
        "skipped_missing_audio": skipped_missing,
        "rows": ingested,
    }


def write_attributions(rows: list[dict], path: Path | None = None) -> Path:
    """Generate ATTRIBUTIONS.md from ingested payload.

    Generated, never hand-maintained: CC-BY requires per-track credit, and a hand-kept list
    silently goes stale the moment the corpus changes.
    """
    path = path or ROOT / "ATTRIBUTIONS.md"
    by_license: dict[str, list[dict]] = {}
    for r in rows:
        by_license.setdefault(r["license_short"], []).append(r)

    lines = [
        "# Attributions",
        "",
        "Every track below is CC0, Public Domain, or CC-BY — no NonCommercial, ShareAlike,",
        "or NoDerivatives terms. This file is generated by `vt ingest`; do not edit it by hand.",
        "",
        "Audio obtained from the [Free Music Archive dataset](https://github.com/mdeff/fma)",
        "(`fma_small`), which distributes per-track license metadata. Track IDs below are FMA",
        "track IDs.",
        "",
        f"**{len(rows)} tracks.**",
        "",
    ]
    for lic in sorted(by_license):
        tracks = sorted(by_license[lic], key=lambda r: (r["artist"].lower(), r["title"].lower()))
        lines += [f"## {lic} ({len(tracks)} tracks)", ""]
        lines += [
            f"- **{r['title']}** — {r['artist']} (FMA track `{r['track_id']}`)" for r in tracks
        ]
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def run(limit: int | None = None, subset: str = "small") -> dict:
    with stage("ingest.metadata") as m:
        rows = corpus.load_fma_metadata(limit=limit, subset=subset)
        m["rows"] = len(rows)

    with stage("ingest.embed_upsert") as m:
        result = ingest_tracks(rows)
        m.update({k: v for k, v in result.items() if k != "rows"})

    with stage("ingest.attributions"):
        write_attributions(result["rows"])

    return result
