"""Generation with four interchangeable backends.

    bank       pre-generated audio, keyed by taste-profile hash. Stage default.
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

Every backend falls back to `bank`. The fallback is loud in the logs and silent on screen:
an audience should never see a stack trace, and the presenter should always know it fired.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import BANK, GEN_BACKEND
from .prompt import GenerationParams

log = logging.getLogger("vector_taste.generate")

BANK_INDEX = BANK / "bank.json"


class GenerationError(RuntimeError):
    pass


@dataclass
class Generated:
    path: Path
    backend: str
    params: GenerationParams
    from_bank: bool
    note: str = ""


def _load_index() -> dict:
    if not BANK_INDEX.exists():
        return {}
    try:
        return json.loads(BANK_INDEX.read_text())
    except json.JSONDecodeError:
        log.warning("bank.json is not valid JSON; treating bank as empty")
        return {}


def _save_index(idx: dict) -> None:
    BANK.mkdir(parents=True, exist_ok=True)
    BANK_INDEX.write_text(json.dumps(idx, indent=2, sort_keys=True))


def bank_lookup(profile_hash: str) -> Path | None:
    """Exact hash match only. Most callers want `bank_best_match` instead."""
    entry = _load_index().get(profile_hash)
    if not entry:
        return None
    path = BANK / entry["file"]
    if not path.exists():
        log.warning("bank entry %s points at missing file %s", profile_hash, path)
        return None
    return path


def bank_best_match(profile_hash: str, centroid=None) -> tuple[Path | None, str]:
    """Exact match if we have it, otherwise the banked taste closest to this one.

    The bank is keyed by taste-profile hash, and a hash changes the moment anyone marks a
    different track. On stage that is *every* interaction — improvising by ear is the whole
    point — so exact-match lookup means the presenter gets silence for any gesture that was
    not baked in advance.

    Falling back to the nearest banked centroid means a live, unrehearsed taste still
    produces musically relevant audio. Returns (path, note); the note is empty on an exact
    hit and names the fallback otherwise, so it can be logged loudly and shown quietly.
    """
    exact = bank_lookup(profile_hash)
    if exact:
        return exact, ""

    if centroid is None:
        return None, ""

    import numpy as np

    best, best_score = None, -2.0
    for h, entry in _load_index().items():
        vec = entry.get("centroid")
        path = BANK / entry["file"]
        if not vec or not path.exists():
            continue
        score = float(np.dot(np.asarray(vec, dtype="float32"), centroid))
        if score > best_score:
            best, best_score = (path, h), score

    if best is None:
        return None, ""
    path, matched_hash = best
    return path, f"nearest banked taste {matched_hash} (cosine {best_score:.3f})"


def bank_add(
    profile_hash: str,
    audio: Path,
    params: GenerationParams,
    backend: str,
    centroid=None,
) -> Path:
    """Register a generated file in the bank so it becomes the instant path next time.

    The centroid is stored so `bank_best_match` can serve an unrehearsed taste from the
    nearest banked one. Without it the bank only answers gestures baked in advance.
    """
    BANK.mkdir(parents=True, exist_ok=True)
    dest = BANK / f"{profile_hash}.wav"
    if audio.resolve() != dest.resolve():
        shutil.copy2(audio, dest)
    idx = _load_index()
    entry = {
        "file": dest.name,
        "prompt": params.prompt,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "duration": params.audio_duration,
        "seed": params.seed,
        "backend": backend,
    }
    if centroid is not None:
        entry["centroid"] = [round(float(x), 6) for x in centroid]
    else:
        # Preserve a centroid recorded by an earlier bake rather than dropping it.
        existing = idx.get(profile_hash, {}).get("centroid")
        if existing:
            entry["centroid"] = existing
    idx[profile_hash] = entry
    _save_index(idx)
    return dest


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
    from .worker import AceStepWorker, WorkerError

    try:
        return AceStepWorker.get().generate(
            job_for(params, profile_hash, out, reference_audio)
        )
    except WorkerError as exc:
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
    centroid=None,
) -> Generated:
    """Generate, or return the closest banked track. Never raises for a missing backend."""
    backend = backend or GEN_BACKEND
    out_dir = out_dir or BANK
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{profile_hash}.wav"

    if backend == "bank":
        banked, note = bank_best_match(profile_hash, centroid)
        if banked:
            if note:
                log.warning("no exact bank entry for %s -- serving %s", profile_hash, note)
            return Generated(banked, "bank", params, from_bank=True, note=note)
        return Generated(
            _placeholder(profile_hash, out_dir),
            "bank",
            params,
            from_bank=True,
            note=(
                f"bank is empty for profile {profile_hash} and no centroid to match on. "
                "Run `vt bake`."
            ),
        )

    try:
        if backend in ("local", "modal"):
            path = _generate_local(params, reference_audio, out, profile_hash)
        elif backend == "replicate":
            path = _generate_replicate(params, out)
        else:
            raise GenerationError(f"unknown GEN_BACKEND {backend!r}")
        bank_add(profile_hash, path, params, backend, centroid=centroid)
        return Generated(path, backend, params, from_bank=False)
    except Exception as exc:  # noqa: BLE001
        # Loud in the logs, quiet on screen.
        log.error("generation backend %r failed: %s -- falling back to bank", backend, exc)
        banked, match_note = bank_best_match(profile_hash, centroid)
        if banked:
            note = f"{backend} failed: {exc}"
            return Generated(
                banked, "bank", params, from_bank=True,
                note=f"{note}; served {match_note}" if match_note else note,
            )
        return Generated(
            _placeholder(profile_hash, out_dir),
            "bank",
            params,
            from_bank=True,
            note=f"{backend} failed and bank is empty: {exc}",
        )


def _placeholder(profile_hash: str, out_dir: Path) -> Path:
    """Silent wav, so the UI always has something to load and never crashes on a 404.

    Deliberately silent rather than a tone: the equalizer will show a flat line, which is an
    honest visual signal that nothing was generated.
    """
    import numpy as np
    import soundfile as sf

    path = out_dir / f"{profile_hash}_placeholder.wav"
    if not path.exists():
        sf.write(path, np.zeros(48_000, dtype="float32"), 48_000)
    return path


def bank_status() -> dict:
    idx = _load_index()
    present = sum(1 for e in idx.values() if (BANK / e["file"]).exists())
    return {"entries": len(idx), "files_present": present, "complete": present == len(idx)}
