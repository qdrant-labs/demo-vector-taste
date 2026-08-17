"""Bake worker. Runs INSIDE ACE-Step's own virtualenv, not this project's.

ACE-Step 1.5 pins transformers<4.58 and CLAP needs >=5, so the two cannot share an
environment. `vt bake` writes a job file and invokes this script with ACE-Step's Python.

All jobs run in one process on purpose. Measured on an M4 Air, the first 30s track costs
~189s but the diffusion steps themselves are only ~1-2s each — the rest is MLX graph
compilation and model offload, both one-time. Spawning a process per track would pay that
cost again for every single job.

Usage (invoked by `vt bake`, not by hand):
    <acestep-venv>/bin/python acestep_bake_worker.py jobs.json

jobs.json: [{"id": str, "caption": str, "duration": float, "bpm": int|null,
             "keyscale": str|null, "seed": int, "reference_audio": str|null,
             "audio_cover_strength": float, "out": str}, ...]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: acestep_bake_worker.py jobs.json", file=sys.stderr)
        return 2

    jobs = json.loads(Path(argv[1]).read_text())
    if not jobs:
        print("no jobs")
        return 0

    # Imported here so the usage error above does not require a 9GB model environment.
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    # cwd is ACE-Step's repo root (set by the caller); checkpoints/ lives beneath it.
    project_root = str(Path.cwd())

    t0 = time.perf_counter()
    print(f"initializing ACE-Step (project_root={project_root})", flush=True)

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=project_root,
        config_path="acestep-v15-turbo",  # 2B turbo: the 8-16GB tier
        device="auto",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
    )
    if not ok:
        print(f"DiT init failed: {status}", file=sys.stderr)
        return 1

    # The LM planner is optional. We supply caption, bpm and keyscale explicitly from
    # retrieved payload, so there is nothing for it to infer — and on a 16GB machine it is
    # the component most likely to push the process into swap.
    llm = LLMHandler()
    print(f"ready in {time.perf_counter() - t0:.1f}s", flush=True)

    failures = 0
    for i, job in enumerate(jobs, 1):
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        print(f"[{i}/{len(jobs)}] {job['id']}: {job['caption'][:56]!r}", flush=True)

        try:
            params = GenerationParams(
                caption=job["caption"][:512],
                lyrics="[Instrumental]",
                instrumental=True,
                bpm=job.get("bpm"),
                keyscale=job.get("keyscale") or "",
                duration=float(job.get("duration", 30.0)),
                inference_steps=int(job.get("inference_steps", 8)),
                seed=int(job.get("seed", 42)),
                task_type="cover" if job.get("reference_audio") else "text2music",
                reference_audio=job.get("reference_audio"),
                audio_cover_strength=float(job.get("audio_cover_strength", 0.7)),
            )
            result = generate_music(dit, llm, params, GenerationConfig(batch_size=1))
            audios = getattr(result, "audios", None) or []
            if not audios:
                raise RuntimeError("generate_music returned no audio")

            src = audios[0].get("path") if isinstance(audios[0], dict) else None
            if src and Path(src).exists():
                shutil.copy2(src, out)
            else:
                import soundfile as sf

                tensor = audios[0]["tensor"]
                sf.write(out, tensor.T.cpu().float().numpy(), audios[0].get("sample_rate", 48000))
            print(f"    -> {out.name}  ({time.perf_counter() - t:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    print(f"\nbaked {len(jobs) - failures}/{len(jobs)} in {time.perf_counter() - t0:.1f}s")
    return 1 if failures == len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
