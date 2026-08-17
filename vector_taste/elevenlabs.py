"""ElevenLabs Music backend — hosted generation in seconds rather than ~2 minutes.

Contract read from the live OpenAPI spec (https://api.elevenlabs.io/openapi.json), not docs
prose, and both endpoints probed against a real key before this was written.

Three constraints from that spec dictate the whole design
---------------------------------------------------------
  1. `seed` CANNOT be combined with `prompt` — it only works with `composition_plan`.
  2. `force_instrumental` CANNOT be combined with `composition_plan` — it is prompt-only.
  3. `music_length_ms` is likewise prompt-only; in plan mode the duration is the sum of the
     chunks' `duration_ms`.

Audio conditioning lives in plan mode, so this backend always uses a composition plan, and
gets instrumental output the only way plan mode allows: the chunk text is "[Instrumental]"
and vocals go in `negative_styles`.

**That makes instrumental a strong hint rather than a guarantee** — worth knowing before a
vocal shows up on stage. Measured on real output it holds (generated tracks score higher
against "instrumental" than against "vocals"), but it is not a flag.

**The seed is NOT reproducible here.** Measured: the same seed produced different audio on
consecutive calls, which matches ElevenLabs' own caveat that "exact reproducibility is not
guaranteed". Different seeds do give different output, so it varies the take, but do not
rely on it to reproduce one. That is why `vt bake` and `vt rehearse` stay on the local
ACE-Step backend, where the seed genuinely is deterministic.

`negative_styles` is a genuine gain over the local backend: it is the first place a rejected
example can be expressed *to the model* rather than only reshaping retrieval.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("vector_taste.elevenlabs")

API = "https://api.elevenlabs.io/v1"
MODEL = os.getenv("ELEVENLABS_MODEL", "music_v2")      # conditioning requires v2; default is v1
OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_48000_192")  # recommended for v2

# The API caps a conditioning reference at 30s, which is exactly our segment length.
MAX_REF_MS = 30_000
CHUNK_MIN_MS, CHUNK_MAX_MS = 3_000, 120_000

# Plan mode has no force_instrumental, so vocals have to be pushed away by description.
NO_VOCALS = ["vocals", "singing", "spoken word", "lyrics", "vocal harmonies", "choir"]


class ElevenLabsError(RuntimeError):
    pass


class _Aborted(Exception):
    """Internal: the user stopped the stream. Translated to GenerationAborted upstream."""


def api_key() -> str | None:
    return os.getenv("ELEVENLABS_API_KEY") or None


def is_available() -> bool:
    """Key present. Does not make a network call."""
    return bool(api_key())


# Uploading bills as a generation, so the same reference clip is uploaded once per process
# and its song_id reused across takes.
_uploads: dict[str, str] = {}
_upload_lock = threading.Lock()


def _styles_from(params, negatives: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Split the synthesized prompt into positive styles, and build negative styles.

    The prompt is already a comma-separated description built from CLAP descriptors, so it
    maps onto `positive_styles` directly rather than needing a second representation.
    """
    positive = [s.strip() for s in params.prompt.split(",") if s.strip()]
    # The API caps these at 50 each.
    positive = [p for p in positive if "no vocals" not in p and p != "instrumental"][:50]

    # A rejected track usually shares traits with the accepted ones -- that is why it
    # surfaced in the same result set. Passing those shared traits as negatives would tell
    # the model to both want and avoid the same thing (measured: a taste built on "acoustic
    # guitar" was sending "acoustic guitar" as a negative). Only genuinely contrasting
    # descriptors are useful here.
    pos_text = " ; ".join(positive).lower()
    contrasting = [
        n for n in (negatives or [])
        if n.lower() not in pos_text and not any(w in pos_text for w in n.lower().split(" and "))
    ]
    negative = list(dict.fromkeys(NO_VOCALS + contrasting))[:50]
    return positive or ["instrumental music"], negative


def upload_reference(path: Path, key: str) -> str | None:
    """Upload a clip and return its song_id, or None if it failed.

    A failed upload must not fail the whole generation: losing style conditioning is much
    better than losing the track.
    """
    import httpx

    cache_key = f"{path.resolve()}:{path.stat().st_mtime_ns}"
    with _upload_lock:
        if cache_key in _uploads:
            return _uploads[cache_key]

    try:
        with path.open("rb") as fh:
            r = httpx.post(
                f"{API}/music/upload",
                headers={"xi-api-key": key},
                files={"file": (path.name, fh, "audio/wav")},
                timeout=180,
            )
        r.raise_for_status()
        song_id = r.json().get("song_id")
    except Exception as exc:  # noqa: BLE001
        log.warning("reference upload failed (%s); generating without conditioning", exc)
        return None

    if song_id:
        with _upload_lock:
            _uploads[cache_key] = song_id
    return song_id


def build_plan(params, song_id: str | None, negatives: list[str] | None = None) -> dict:
    """One-chunk composition plan. Plan mode is required for seed AND conditioning."""
    positive, negative = _styles_from(params, negatives)
    duration = int(
        min(max(params.audio_duration * 1000, CHUNK_MIN_MS), CHUNK_MAX_MS)
    )
    chunk: dict = {
        "text": "[Instrumental]",       # the only lever for instrumental in plan mode
        "duration_ms": duration,
        "positive_styles": positive,
        "negative_styles": negative,
        "context_adherence": "high",
    }
    if song_id:
        chunk["conditioning_ref"] = {
            "song_id": song_id,
            "range": {"start_ms": 0, "end_ms": min(duration, MAX_REF_MS)},
        }
        chunk["condition_strength"] = os.getenv("ELEVENLABS_CONDITION_STRENGTH", "medium")
    return {"chunks": [chunk]}


def compose(params, out: Path, reference_audio: Path | None = None,
            negatives: list[str] | None = None) -> Path:
    """Generate a track. Synchronous: the response body IS the audio.

    Registers an aborter so the Stop button works on this backend too -- without it, Stop
    would silently do nothing here, since there is no process to kill.
    """
    import httpx

    from .progress import PROGRESS, register_aborter

    key = api_key()
    if not key:
        raise ElevenLabsError(
            "ELEVENLABS_API_KEY is not set. Add it to .env.local, or switch the generator "
            "to Local."
        )

    PROGRESS.reset("preparing")
    PROGRESS.update(backend="elevenlabs", desc="uploading reference", phase="preparing")

    song_id = upload_reference(reference_audio, key) if reference_audio else None
    plan = build_plan(params, song_id, negatives)

    body = {"model_id": MODEL, "seed": int(params.seed), "composition_plan": plan}

    # The response is STREAMED so abort can actually take effect. Closing an httpx client
    # from another thread does not interrupt an in-flight request -- measured: abort
    # returned True and the generation completed and wrote a file anyway. Checking a flag
    # between chunks is the only thing that genuinely stops the wait.
    cancel = threading.Event()
    register_aborter(lambda: (cancel.set(), True)[1])
    PROGRESS.update(
        phase="generating",
        desc="generating" + (" (style-conditioned)" if song_id else ""),
    )

    chunks: list[bytes] = []
    try:
        with httpx.Client(timeout=300) as client, client.stream(
            "POST",
            f"{API}/music",
            params={"output_format": OUTPUT_FORMAT},
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json=body,
        ) as r:
            if r.status_code != 200:
                r.read()
                if r.status_code == 401:
                    raise ElevenLabsError("ELEVENLABS_API_KEY was rejected (401)")
                if r.status_code == 429:
                    raise ElevenLabsError("rate limited or out of credits (429)")
                raise ElevenLabsError(f"HTTP {r.status_code}: {r.text[:300]}")
            for chunk in r.iter_bytes():
                if cancel.is_set():
                    raise _Aborted
                chunks.append(chunk)
    except _Aborted:
        # Nothing is written: the user asked for silence, not a partial file.
        log.warning("elevenlabs generation aborted by user")
        raise
    except httpx.HTTPError as exc:
        raise ElevenLabsError(f"request failed: {exc}") from exc
    finally:
        register_aborter(None)

    audio = b"".join(chunks)
    if not audio:
        raise ElevenLabsError("empty response body")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)
    log.info("elevenlabs generated %s (%d bytes)", out.name, len(audio))
    return out
