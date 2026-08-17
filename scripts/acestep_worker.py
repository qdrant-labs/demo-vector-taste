"""Resident ACE-Step worker. Runs INSIDE ACE-Step's own virtualenv, not this project's.

ACE-Step 1.5 pins transformers<4.58 and CLAP needs >=5, so the two cannot share an
environment. This process is spawned with ACE-Step's interpreter and kept alive.

Why resident rather than one process per track
----------------------------------------------
Measured on an M4 Air: a 30s track takes ~120s, but the diffusion steps themselves run
1-2s each. Nearly all of it is model load plus MLX graph compilation, both ONE-TIME. A
process per request pays that every time; staying resident pays it once. Both `vt bake`
(feeds N jobs, closes stdin) and live generation (keeps stdin open) use this same worker,
so the batch and interactive paths cannot drift apart.

Protocol
--------
stdin  : one JSON job per line.
stdout : one `@@VT@@ {json}` line per event.

The sentinel matters. ACE-Step and its dependencies write freely to stdout and stderr
(loguru, tqdm progress bars, torch warnings), so the parent cannot assume stdout is clean
JSON. It scans for the prefix instead.

Events: {"event":"ready"} once models are loaded, then {"event":"result", "id":..,
"ok":bool, "path":str|null, "error":str|null, "seconds":float} per job.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

SENTINEL = "@@VT@@"


def emit(**payload) -> None:
    """Write one protocol line. Flushed immediately so the parent isn't left waiting."""
    sys.stdout.write(f"{SENTINEL} {json.dumps(payload)}\n")
    sys.stdout.flush()


def run_job(dit, llm, job: dict) -> dict:
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    params = GenerationParams(
        caption=job["caption"][:512],          # ACE-Step caps the caption at 512 chars
        lyrics="[Instrumental]",
        instrumental=True,
        bpm=job.get("bpm"),
        keyscale=job.get("keyscale") or "",
        duration=float(job.get("duration", 30.0)),
        inference_steps=int(job.get("inference_steps", 8)),
        seed=int(job.get("seed", 42)),
        # text2music, NOT cover, even with a style reference:
        #   reference_audio -> timbre/style conditioning, works with text2music
        #   src_audio       -> the track being re-recorded, what `cover` requires
        # Passing reference_audio with task_type="cover" fails instantly with
        # "Task 'cover' requires source audio".
        task_type="text2music",
        reference_audio=job.get("reference_audio"),
        audio_cover_strength=float(job.get("audio_cover_strength", 0.7)),
    )

    result = generate_music(dit, llm, params, GenerationConfig(batch_size=1))
    audios = getattr(result, "audios", None) or []
    if not audios:
        # GenerationResult carries the real reason in .error / .status_message. Reporting
        # only "no audio" hides it -- that cost four identical silent failures once.
        raise RuntimeError(
            f"no audio (success={getattr(result, 'success', '?')}) "
            f"error={getattr(result, 'error', None)!r} "
            f"status={getattr(result, 'status_message', '')!r}"
        )

    src = audios[0].get("path") if isinstance(audios[0], dict) else None
    if src and Path(src).exists():
        shutil.copy2(src, out)
    else:
        import soundfile as sf

        sf.write(out, audios[0]["tensor"].T.cpu().float().numpy(),
                 audios[0].get("sample_rate", 48000))

    return {"path": str(out), "seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    t0 = time.perf_counter()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(Path.cwd()),      # ACE-Step resolves checkpoints/ relative to cwd
        config_path="acestep-v15-turbo",   # 2B turbo: the 8-16GB tier
        device="auto",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
    )
    if not ok:
        emit(event="ready", ok=False, error=f"DiT init failed: {status}")
        return 1

    # The LM planner is optional: caption, bpm and keyscale are supplied explicitly from
    # retrieved payload, so there is nothing for it to infer -- and on a 16GB machine it is
    # the component most likely to push the process into swap.
    llm = LLMHandler()
    emit(event="ready", ok=True, seconds=round(time.perf_counter() - t0, 1))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            emit(event="result", id=None, ok=False, error=f"bad job json: {exc}")
            continue
        try:
            emit(event="result", id=job.get("id"), ok=True, **run_job(dit, llm, job))
        except Exception as exc:  # noqa: BLE001 - one bad job must not kill the worker
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            emit(event="result", id=job.get("id"), ok=False,
                 error=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
