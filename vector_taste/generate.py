"""Generation with three interchangeable backends.

    elevenlabs hosted, seconds per track. The default.
    local      ACE-Step 1.5 via a resident worker process (Apple Silicon MLX, or CUDA)
    modal      your own Modal deployment
    replicate  hosted, one API token

Why ACE-Step runs out of process
--------------------------------
ACE-Step 1.5 pins `transformers>=4.51,<4.58`; CLAP here needs `transformers>=5`. They
cannot share a virtualenv. So ACE-Step is installed separately (see
scripts/acestep_setup.sh) and driven by a resident subprocess (see worker.py) that keeps
the model loaded between requests -- loading it costs ~60-100s while the diffusion steps
themselves are 1-2s each. Fully offline: no sockets, no network.

There is deliberately NO fallback. A backend that fails raises, and the caller reports it.
An earlier design quietly served pre-baked audio instead, which meant a broken generator was
indistinguishable from a working one -- on stage that is worse than a visible error, because
you find out afterwards.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .config import GEN_BACKEND, GENERATED
from .prompt import GenerationParams

log = logging.getLogger("vector_taste.generate")


class GenerationError(RuntimeError):
    pass


class GenerationAborted(GenerationError):
    """The user stopped it. Distinguished from a real failure so the route can answer 499
    rather than reporting an error nobody hit."""


@dataclass
class Generated:
    path: Path
    backend: str
    params: GenerationParams
    note: str = ""


def latest_generated(profile_hash: str) -> Path | None:
    """Most recent live generation for a taste, or None.

    Live output is named `<hash>-<seed>.<ext>`, so this is how the loop and the UI find the
    take the user just heard.
    """
    if not GENERATED.exists():
        return None
    takes = sorted(
        (p for ext in ("wav", "mp3") for p in GENERATED.glob(f"{profile_hash}-*.{ext}")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return takes[0] if takes else None


def audio_for_profile(profile_hash: str) -> tuple[Path | None, str]:
    """The audio a user most recently heard for this taste.

    NOT a fallback: this is how the loop scores the track that was just composed. When
    nothing has been generated for a taste there is genuinely nothing to score, and callers
    say so rather than substituting something older.
    """
    return latest_generated(profile_hash), ""


# Which backends can sing. Instrumental is enforced backend-side everywhere else: ACE-Step
# is driven with instrumental=True and has no lyrics source, so offering vocals there would
# produce wordless syllables and misrepresent what it does.
VOCALS_BACKENDS = frozenset({"elevenlabs"})

# Backends that transmit the style-reference CLIP to a third party. Distinct from
# VOCALS_BACKENDS on purpose even though they hold the same value today: one is about what a
# backend can produce, this is about whose disk the audio ends up on. `modal` is your own
# deployment and `replicate` never receives a reference at all.
AUDIO_LEAVES_MACHINE = frozenset({"elevenlabs"})


def available_backends() -> dict[str, bool]:
    """Which backends can run, so the UI can disable rather than fail on click."""
    from .elevenlabs import is_available as el_available
    from .worker import is_available as local_available

    return {
        "local": local_available(),
        "elevenlabs": el_available(),
        "replicate": bool(os.getenv("REPLICATE_API_TOKEN")),
    }


def job_for(params: GenerationParams, profile_hash: str, out: Path,
            reference_audio: Path | None = None) -> dict:
    """Worker job dict. Shared by live generation and `vt bake` so they cannot diverge."""
    return {
        "id": profile_hash,
        "caption": params.prompt,
        "duration": params.audio_duration,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "seed": params.seed,
        "inference_steps": params.num_inference_steps,
        "reference_audio": str(reference_audio) if reference_audio else None,
        "audio_cover_strength": params.audio_cover_strength,
        "out": str(out),
    }


def _generate_local(
    params: GenerationParams, reference_audio: Path | None, out: Path, profile_hash: str
) -> Path:
    """Generate via the resident ACE-Step worker.

    This replaces an HTTP POST to `ACESTEP_URL/generate`, an endpoint that does not exist --
    ACE-Step's API is task-based (/create_random_sample -> /query_result), so that path had
    never actually worked. Talking to our own worker is both simpler and under our control.
    """
    from .worker import AceStepWorker, WorkerAborted, WorkerError

    try:
        return AceStepWorker.get().generate(
            job_for(params, profile_hash, out, reference_audio)
        )
    except WorkerAborted as exc:
        raise GenerationAborted(str(exc)) from exc
    except WorkerError as exc:
        raise GenerationError(str(exc)) from exc


def _generate_elevenlabs(
    params: GenerationParams,
    reference_audio: Path | None,
    out: Path,
    negatives: list[str] | None = None,
    vocals: bool = False,
) -> Path:
    """Hosted generation in seconds. Supports a real seed AND style conditioning."""
    from .elevenlabs import ElevenLabsError, _Aborted, compose

    try:
        return compose(params, out, reference_audio, negatives, vocals)
    except _Aborted as exc:
        raise GenerationAborted("generation aborted") from exc
    except ElevenLabsError as exc:
        raise GenerationError(str(exc)) from exc


def _generate_replicate(params: GenerationParams, out: Path) -> Path:
    """Hosted fallback. Note: does NOT support audio style reference."""
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise GenerationError("REPLICATE_API_TOKEN is not set")
    try:
        import replicate
    except ImportError as exc:
        raise GenerationError("pip install 'vector-taste[replicate]'") from exc

    import httpx

    output = replicate.run(
        "lucataco/ace-step",
        input={
            "tags": params.prompt,
            "lyrics": params.lyrics or "[instrumental]",
            "duration": int(params.audio_duration),
            "seed": params.seed,
        },
    )
    url = str(output[0] if isinstance(output, list) else output)
    out.write_bytes(httpx.get(url, timeout=300).content)
    return out


def generate(
    params: GenerationParams,
    profile_hash: str,
    reference_audio: Path | None = None,
    backend: str | None = None,
    out_dir: Path | None = None,
    negatives: list[str] | None = None,
    vocals: bool = False,
) -> Generated:
    """Generate a track, or raise GenerationError saying why it could not."""
    backend = backend or GEN_BACKEND
    if vocals and backend not in VOCALS_BACKENDS:
        # Silently dropping this would hand back an instrumental with no hint that the
        # request was ignored. Callers check VOCALS_BACKENDS first.
        raise GenerationError(f"backend {backend!r} cannot generate vocals")
    # Each compose is a new file named by its seed, so composing the same taste twice keeps
    # both takes rather than overwriting one with the other.
    out_dir = out_dir or GENERATED
    out_dir.mkdir(parents=True, exist_ok=True)
    # ElevenLabs returns mp3; naming those .wav would be a lie that breaks the media type
    # and confuses anything that sniffs by extension.
    ext = "mp3" if backend == "elevenlabs" else "wav"
    out = out_dir / f"{profile_hash}-{params.seed}.{ext}"

    try:
        if backend in ("local", "modal"):
            path = _generate_local(params, reference_audio, out, profile_hash)
        elif backend == "elevenlabs":
            path = _generate_elevenlabs(params, reference_audio, out, negatives, vocals)
        elif backend == "replicate":
            path = _generate_replicate(params, out)
        else:
            raise GenerationError(f"unknown GEN_BACKEND {backend!r}")
        return Generated(path, backend, params)
    except GenerationAborted:
        raise                              # a deliberate stop is not a failure
    except GenerationError:
        raise                              # already carries a message worth showing
    except Exception as exc:  # noqa: BLE001 - one place to turn any backend fault into ours
        log.error("generation backend %r failed: %s", backend, exc)
        raise GenerationError(f"{backend} failed: {exc}") from exc
