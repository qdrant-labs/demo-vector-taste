"""Close the loop: re-embed the generated track and measure where it landed.

This is how the talk ends, so the number has to be honest and stable.

The generated audio is chunked into 10s windows and embedded with the SAME CLAP model as
the corpus — never as one whole-file embedding, which would hit CLAP's random 10s crop and
make the closing number different on every run.

The headline is a PERCENTILE, not a bare cosine. A cosine of 0.61 means nothing to an
audience and looks identical whether the demo worked or not. "Closer to your taste than 94%
of the library" is falsifiable and legible. Raw cosine is reported alongside it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import store
from .embed import embed_audio_file
from .taste import TasteProfile, percentile_against_centroid, recommend, taste_centroid

_NS = uuid.UUID("5f3d9a1e-0001-4000-8000-000000000000")


@dataclass
class LoopResult:
    cosine: float
    percentile: float
    population: int
    baseline_cosine: float
    baseline_percentile: float
    n_chunks: int
    run_id: str

    @property
    def beats_baseline(self) -> bool:
        return self.cosine >= self.baseline_cosine

    def summary(self) -> str:
        verdict = (
            "as close as the closest human track"
            if self.beats_baseline
            else "not as close as the closest human track"
        )
        return "\n".join(
            [
                "",
                f"  The generated track lands at the {self.percentile:.0f}th percentile.",
                f"  Closer to your taste centroid than {self.percentile:.0f}% of the "
                f"{self.population} segments in the library.",
                "",
                f"    generated   cosine {self.cosine:+.4f}   percentile {self.percentile:5.1f}",
                f"    best human  cosine {self.baseline_cosine:+.4f}   "
                f"percentile {self.baseline_percentile:5.1f}   <- retrieval baseline",
                "",
                f"  Verdict: {verdict}.",
                "",
            ]
        )


def close_loop(
    audio_path: Path,
    profile: TasteProfile,
    upsert: bool = True,
    run_id: str | None = None,
) -> LoopResult:
    """Embed the generated track, score it against the taste centroid, optionally store it."""
    run_id = run_id or f"gen-{profile.hash}"
    centroid_vec = taste_centroid(profile)

    vecs = embed_audio_file(audio_path)
    if not len(vecs):
        raise ValueError(f"no audio embedded from {audio_path}")

    # Score the generated track by its BEST chunk, the same max-sim rule used for corpus
    # segments — otherwise we would be comparing it on different terms than everything else.
    sims = vecs @ centroid_vec
    best_i = int(np.argmax(sims))
    cos, pct, population = percentile_against_centroid(vecs[best_i], centroid_vec)

    # Baseline: the best human track that is NOT one of the user's own examples.
    #
    # Excluding the positives matters. The centroid is the mean OF those positives, so a
    # positive scores ~0.91 against it by construction — comparing generation to that is
    # comparing it to the question rather than to an answer, and makes any result look bad.
    # The fair question is "how close did the machine get, versus the closest thing the
    # library already had that you hadn't already picked?"
    chosen = set(profile.positives)
    top = [h for h in recommend(profile, limit=10) if h.point_id not in chosen]
    if top:
        b_cos, b_pct, _ = percentile_against_centroid(_fetch_one(top[0].point_id), centroid_vec)
    else:
        b_cos, b_pct = float("nan"), float("nan")

    if upsert:
        store.ensure_collection()
        points = []
        for i, vec in enumerate(vecs):
            points.append(
                store.make_point(
                    str(uuid.uuid5(_NS, f"{run_id}:{i}")),
                    vec,
                    {
                        "track_id": run_id,
                        "segment_id": f"{run_id}:0",
                        "segment_index": 0,
                        "chunk_index": i,
                        "artist": "Vector Taste",
                        "title": f"Generated ({profile.hash})",
                        "start_sec": i * 10,
                        "end_sec": (i + 1) * 10,
                        "bpm": None,
                        "key": None,
                        "tags": ["generated"],
                        "caption": "generated track",
                        "license": "generated",
                        "source_url": "",
                        # The flag every demo query filters on. Without it these points
                        # become search neighbours and shift the number on re-runs.
                        "is_generated": True,
                        "generation_run_id": run_id,
                        "audio_path": str(audio_path),
                    },
                )
            )
        store.upsert_points(points)

    return LoopResult(
        cosine=cos,
        percentile=pct,
        population=population,
        baseline_cosine=b_cos,
        baseline_percentile=b_pct,
        n_chunks=len(vecs),
        run_id=run_id,
    )


def _fetch_one(point_id: str) -> np.ndarray:
    from .search import fetch_vectors

    vecs = fetch_vectors([point_id])
    if point_id not in vecs:
        raise ValueError(f"point {point_id} not found")
    return vecs[point_id]
