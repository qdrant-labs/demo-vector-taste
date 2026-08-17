"""Pre-generate the bank.

The bank is the stage path: generation is instant and cannot fail in front of an audience.

Baking shells out to `scripts/acestep_bake_worker.py` running under ACE-Step's own Python,
because ACE-Step pins transformers<4.58 while CLAP here needs >=5. All profiles are baked
in ONE worker process: on an M4 Air the first 30s track costs ~189s but the diffusion steps
are only ~1-2s each, so nearly all of it is one-time MLX compilation and model load.

Bake wherever you like — the bank is just audio files, so hardware does not affect the demo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import BANK, DATA, ROOT
from .generate import bank_add
from .prompt import load as load_prompt
from .prompt import save as save_prompt
from .prompt import synthesize
from .taste import TasteProfile, recommend
from .timing import stage

WORKER = ROOT / "scripts" / "acestep_bake_worker.py"


def acestep_dir() -> Path:
    """ACE-Step's repo root. Its `acestep` package is importable from here."""
    # Guard the env var: Path("") / ".venv" collapses to a RELATIVE ".venv/bin/python",
    # which matches THIS project's interpreter and silently bakes with the wrong env.
    env = os.getenv("ACESTEP_DIR")
    return Path(env).expanduser().resolve() if env else (ROOT / ".acestep").resolve()


def acestep_python() -> Path | None:
    py = acestep_dir() / ".venv" / "bin" / "python"
    return py if py.is_file() else None


def discover_profiles() -> list[str]:
    """Every saved taste profile is a bank candidate.

    Profiles are saved under both a friendly name and their hash; the bank is keyed by hash,
    so dedupe to the hashes actually referenced.
    """
    hashes = []
    for path in sorted(DATA.glob("taste_*.json")):
        try:
            hashes.append(json.loads(path.read_text())["hash"])
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(set(hashes))


def build_jobs(profile_hashes: list[str], duration: float) -> list[dict]:
    """Turn taste profiles into worker jobs, caching each synthesized prompt to disk."""
    from .cli import _reference_clip

    jobs = []
    for h in profile_hashes:
        profile = TasteProfile.load(h)
        hits = recommend(profile, limit=10)
        synth = load_prompt(h) or synthesize(hits, steer=profile.steer, duration=duration)
        save_prompt(h, synth)

        ref = _reference_clip(hits[0]) if hits else None
        p = synth.params
        jobs.append(
            {
                "id": h,
                "caption": p.prompt,
                "duration": p.audio_duration,
                "bpm": p.bpm,
                "keyscale": p.keyscale,
                "seed": p.seed,
                "inference_steps": p.num_inference_steps,
                "reference_audio": str(ref) if ref else None,
                "audio_cover_strength": p.audio_cover_strength,
                "out": str(BANK / f"{h}.wav"),
            }
        )
    return jobs


def bake_bank(
    profiles: list[str] | None = None, backend: str = "local", duration: float = 30.0
) -> dict:
    targets = profiles or discover_profiles()
    if not targets:
        print("  no taste profiles found. Run `vt demo-profiles` first.")
        return {"baked": 0, "failed": 0}

    py = acestep_python()
    if py is None:
        print("  ACE-Step is not installed. Run:  ./scripts/acestep_setup.sh")
        print("  (or bake elsewhere and drop the .wav files into bank/)")
        return {"baked": 0, "failed": len(targets)}

    jobs = build_jobs(targets, duration)
    BANK.mkdir(parents=True, exist_ok=True)
    job_file = BANK / "_jobs.json"
    job_file.write_text(json.dumps(jobs, indent=2))

    print(f"  baking {len(jobs)} profile(s) with {py}")
    for j in jobs:
        print(f"    {j['id']}  {j['caption'][:60]}")
    print()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    with stage("bake.all", profiles=len(jobs)) as m:
        proc = subprocess.run(  # noqa: S603
            [str(py), str(WORKER), str(job_file)],
            # ACE-Step resolves both its `acestep` package and its checkpoints/ directory
            # relative to cwd, so this must be its repo root, not ours.
            cwd=str(acestep_dir()),
            env=env,
            check=False,
        )
        m["returncode"] = proc.returncode

    baked = failed = 0
    for j in jobs:
        out = Path(j["out"])
        if out.exists() and out.stat().st_size > 1024:
            synth = load_prompt(j["id"])
            if synth:
                bank_add(j["id"], out, synth.params, backend)
            baked += 1
        else:
            failed += 1
            print(f"  {j['id']}: no audio produced")

    job_file.unlink(missing_ok=True)
    print(f"\n  done: {baked} baked, {failed} failed  ->  {BANK}")
    return {"baked": baked, "failed": failed}


def import_bank(src_dir: Path) -> int:
    """Adopt .wav files baked elsewhere (a GPU box, a colleague's machine).

    Filenames must be <profile_hash>.wav; the bank is keyed by hash so a mislabelled file
    would silently never be found.
    """
    n = 0
    for wav in sorted(Path(src_dir).glob("*.wav")):
        synth = load_prompt(wav.stem)
        if synth is None:
            print(f"  skip {wav.name}: no cached prompt for that profile hash")
            continue
        shutil.copy2(wav, BANK / wav.name)
        bank_add(wav.stem, BANK / wav.name, synth.params, "imported")
        n += 1
    print(f"  imported {n} track(s)")
    return n


if __name__ == "__main__":
    sys.exit(0 if bake_bank()["failed"] == 0 else 1)
