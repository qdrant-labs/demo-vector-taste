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

Vocals
------
In plan mode a chunk's `text` IS the lyric content, and this demo retrieves sound, not
words — there are no lyrics to put there. So `vocals=True` asks `POST /v1/music/plan` to
write a plan from our synthesized prompt, then keeps only its `text` and overlays our own
styles, negatives, durations and `conditioning_ref`. The taste stays ours; ElevenLabs
supplies the words. If that call fails the track still generates, from a bare "[Verse]"
marker plus a "sung vocals" style.

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


def _styles_from(
    params, negatives: list[str] | None = None, vocals: bool = False
) -> tuple[list[str], list[str]]:
    """Split the synthesized prompt into positive styles, and build negative styles.

    The prompt is already a comma-separated description built from CLAP descriptors, so it
    maps onto `positive_styles` directly rather than needing a second representation.
    """
    positive = [s.strip() for s in params.prompt.split(",") if s.strip()]
    # The API caps these at 50 each.
    positive = [p for p in positive if "no vocals" not in p and p != "instrumental"]
    if vocals:
        positive.append("sung vocals")
    positive = positive[:50]

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
    # NO_VOCALS is how instrumental is enforced in plan mode, so it has to go when the user
    # asked to hear a voice -- otherwise the request wants and forbids singing at once.
    base = [] if vocals else NO_VOCALS
    negative = list(dict.fromkeys(base + contrasting))[:50]
    default = ["vocal music"] if vocals else ["instrumental music"]
    return positive or default, negative


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


def plan_with_lyrics(params, key: str) -> list[dict] | None:
    """Ask ElevenLabs to write a plan -- lyrics included -- and return its chunks.

    In plan mode a chunk's `text` IS the lyric content, and we have CLAP descriptors rather
    than words. This endpoint is the only way to get real lines while keeping the seed and
    audio conditioning that only plan mode offers.

    Returns None on any failure. A missing set of lyrics must not cost the whole track --
    build_plan() falls back to a bare section marker, same rule as upload_reference().
    """
    import httpx

    try:
        r = httpx.post(
            f"{API}/music/plan",
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={
                # The planner follows the prompt, and ours describes SOUND -- descriptors
                # from CLAP, nothing about a voice. Measured: it answered "[Intro]" and
                # "[Groove]" with no words at all. Asking for lyrics outright is what makes
                # this endpoint return any.
                "prompt": f"{params.prompt}. A song with sung vocals: "
                          f"write lyrics for every section."[:4100],
                "music_length_ms": _clamp_ms(params.audio_duration),
                "model_id": MODEL,
            },
            timeout=120,
        )
        r.raise_for_status()
        chunks = r.json().get("chunks")
    except Exception as exc:  # noqa: BLE001
        log.warning("lyric plan failed (%s); using a bare section marker", exc)
        return None

    # music_v1 answers with the older `sections` shape instead of `chunks`. Anything without
    # chunks is not a plan we can overlay onto, so treat it as a miss rather than guessing.
    if not isinstance(chunks, list) or not chunks:
        log.warning("lyric plan returned no chunks; using a bare section marker")
        return None
    return chunks


def _clamp_ms(seconds: float) -> int:
    return int(min(max(seconds * 1000, CHUNK_MIN_MS), CHUNK_MAX_MS))


def build_plan(
    params,
    song_id: str | None,
    negatives: list[str] | None = None,
    vocals: bool = False,
    lyric_chunks: list[dict] | None = None,
) -> dict:
    """Composition plan. Plan mode is required for seed AND conditioning.

    Instrumental is one chunk saying so. Vocals keep the TEXT of whatever ElevenLabs wrote
    (that is the lyrics) while everything else -- styles, negatives, durations, conditioning
    -- is ours, because the taste is the thing being demonstrated, not their plan.
    """
    positive, negative = _styles_from(params, negatives, vocals)
    duration = _clamp_ms(params.audio_duration)

    if vocals:
        texts = [c.get("text") or "[Verse]" for c in (lyric_chunks or [])] or ["[Verse]"]
    else:
        texts = ["[Instrumental]"]      # the only lever for instrumental in plan mode

    # Split our duration across however many chunks came back, so a multi-section plan does
    # not silently multiply the length we asked for.
    each = max(CHUNK_MIN_MS, duration // len(texts))
    chunks: list[dict] = [
        {
            "text": text[:6000],
            "duration_ms": min(each, CHUNK_MAX_MS),
            "positive_styles": positive,
            "negative_styles": negative,
            "context_adherence": "high",
        }
        for text in texts
    ]

    if song_id:
        # First chunk only: the spec says it "will influence the generation of all
        # subsequent chunks", so repeating the reference buys nothing.
        chunks[0]["conditioning_ref"] = {
            "song_id": song_id,
            "range": {"start_ms": 0, "end_ms": min(chunks[0]["duration_ms"], MAX_REF_MS)},
        }
        chunks[0]["condition_strength"] = os.getenv("ELEVENLABS_CONDITION_STRENGTH", "medium")
    return {"chunks": chunks}


def compose(params, out: Path, reference_audio: Path | None = None,
            negatives: list[str] | None = None, vocals: bool = False) -> Path:
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

    lyric_chunks = None
    if vocals:
        PROGRESS.update(desc="writing lyrics")
        lyric_chunks = plan_with_lyrics(params, key)
    plan = build_plan(params, song_id, negatives, vocals, lyric_chunks)

    body = {"model_id": MODEL, "seed": int(params.seed), "composition_plan": plan}

    # The response is STREAMED so abort can actually take effect. Closing an httpx client
    # from another thread does not interrupt an in-flight request -- measured: abort
    # returned True and the generation completed and wrote a file anyway. Checking a flag
    # between chunks is the only thing that genuinely stops the wait.
    cancel = threading.Event()
    register_aborter(lambda: (cancel.set(), True)[1])
    PROGRESS.update(
        phase="generating",
        desc="generating"
        + (" with vocals" if vocals else "")
        + (" (style-conditioned)" if song_id else ""),
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
