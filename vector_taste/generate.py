"""Generation with four interchangeable backends.

    bank       pre-generated audio, keyed by taste-profile hash. Stage default.
    local      ACE-Step 1.5 via its own REST API (Apple Silicon MLX, or CUDA)
    modal      your own Modal deployment
    replicate  hosted, one API token

Why ACE-Step runs out of process
--------------------------------
ACE-Step 1.5 pins `transformers>=4.51,<4.58`; CLAP here needs `transformers>=5`. They
cannot share a virtualenv. So ACE-Step is installed separately (see
scripts/acestep_setup.sh) and spoken to over its own localhost REST API. That is still
fully offline — localhost is not the network — and it keeps the model resident between
requests, which matters because loading it is far slower than generating with it.

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

ACESTEP_URL = os.getenv("ACESTEP_URL", "http://localhost:8001")
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
    entry = _load_index().get(profile_hash)
    if not entry:
        return None
    path = BANK / entry["file"]
    if not path.exists():
        log.warning("bank entry %s points at missing file %s", profile_hash, path)
        return None
    return path


def bank_add(profile_hash: str, audio: Path, params: GenerationParams, backend: str) -> Path:
    """Register a generated file in the bank so it becomes the instant path next time."""
    BANK.mkdir(parents=True, exist_ok=True)
    dest = BANK / f"{profile_hash}.wav"
    if audio.resolve() != dest.resolve():
        shutil.copy2(audio, dest)
    idx = _load_index()
    idx[profile_hash] = {
        "file": dest.name,
        "prompt": params.prompt,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "duration": params.audio_duration,
        "seed": params.seed,
        "backend": backend,
    }
    _save_index(idx)
    return dest


def _generate_local(params: GenerationParams, reference_audio: Path | None, out: Path) -> Path:
    """POST to a locally running ACE-Step API server."""
    import httpx

    body: dict = {
        "prompt": params.prompt,
        "lyrics": params.lyrics,
        "audio_duration": params.audio_duration,
        "infer_step": params.num_inference_steps,
        "shift": params.shift,
        "seed": params.seed,
        "task_type": "cover" if reference_audio else "text2music",
    }
    if params.bpm:
        body["bpm"] = params.bpm
    if params.keyscale:
        body["keyscale"] = params.keyscale
    if reference_audio:
        body["reference_audio"] = str(reference_audio)
        body["audio_cover_strength"] = params.audio_cover_strength

    try:
        r = httpx.post(f"{ACESTEP_URL}/generate", json=body, timeout=900)
        r.raise_for_status()
    except Exception as exc:
        raise GenerationError(
            f"ACE-Step API at {ACESTEP_URL} unreachable or failed ({exc}). "
            "Start it with scripts/acestep_setup.sh, or use GEN_BACKEND=bank."
        ) from exc

    ctype = r.headers.get("content-type", "")
    if ctype.startswith("audio/") or ctype == "application/octet-stream":
        out.write_bytes(r.content)
        return out

    # Otherwise the server returns JSON pointing at a file it wrote.
    data = r.json()
    src = data.get("path") or data.get("audio_path") or (data.get("audios") or [None])[0]
    if not src or not Path(src).exists():
        raise GenerationError(f"ACE-Step returned no usable audio path: {str(data)[:200]}")
    shutil.copy2(src, out)
    return out


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
) -> Generated:
    """Generate, or return the banked track. Never raises for a missing backend."""
    backend = backend or GEN_BACKEND
    out_dir = out_dir or BANK
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{profile_hash}.wav"

    if backend == "bank":
        banked = bank_lookup(profile_hash)
        if banked:
            return Generated(banked, "bank", params, from_bank=True)
        return Generated(
            _placeholder(profile_hash, out_dir),
            "bank",
            params,
            from_bank=True,
            note=(
                f"no bank entry for profile {profile_hash}; emitted silence. "
                "Bake one with `vt bake`."
            ),
        )

    try:
        if backend in ("local", "modal"):
            path = _generate_local(params, reference_audio, out)
        elif backend == "replicate":
            path = _generate_replicate(params, out)
        else:
            raise GenerationError(f"unknown GEN_BACKEND {backend!r}")
        bank_add(profile_hash, path, params, backend)
        return Generated(path, backend, params, from_bank=False)
    except Exception as exc:  # noqa: BLE001
        # Loud in the logs, silent on screen.
        log.error("generation backend %r failed: %s -- falling back to bank", backend, exc)
        banked = bank_lookup(profile_hash)
        if banked:
            return Generated(banked, "bank", params, from_bank=True, note=f"{backend} failed: {exc}")
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
