"""Pre-generate the bank.

The bank is the stage path: generation is instant and cannot fail in front of an audience.

Baking uses the same resident worker as live generation (`vector_taste/worker.py`), so the
batch and interactive paths cannot drift apart. All profiles go through one worker process:
on an M4 Air a 30s track costs ~120s but the diffusion steps are only 1-2s each, so nearly
all of it is one-time MLX compilation and model load.

Bake wherever you like — the bank is just audio files, so hardware does not affect the demo.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .config import BANK, DATA
from .generate import bank_add, job_for
from .prompt import save as save_prompt
from .prompt import seed_from_hash, synthesize
from .taste import TasteProfile, negative_hits, recommend, taste_centroid
from .timing import stage
from .worker import AceStepWorker, WorkerError, acestep_dir


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


def _synth_for(profile_hash: str, duration: float, steps: int):
    """Synthesize params for a profile. One place, so bake and re-bake cannot disagree.

    Always re-synthesized rather than reusing the cached prompt: a cache may predate the
    descriptor vocabulary or the seed derivation, and baking stale params is exactly how
    the bank ended up sounding identical across profiles.
    """
    profile = TasteProfile.load(profile_hash)
    hits = recommend(profile, limit=10)
    synth = synthesize(
        hits, steer=profile.steer, duration=duration, steps=steps,
        negatives=negative_hits(profile), seed=seed_from_hash(profile_hash),
    )
    return profile, hits, synth


def build_jobs(profile_hashes: list[str], duration: float, steps: int = 8) -> list[dict]:
    """Turn taste profiles into worker jobs, caching each synthesized prompt to disk."""
    from .cli import _reference_clip

    jobs = []
    for h in profile_hashes:
        _profile, hits, synth = _synth_for(h, duration, steps)
        save_prompt(h, synth)
        ref = _reference_clip(hits[0]) if hits else None
        jobs.append(job_for(synth.params, h, BANK / f"{h}.wav", ref))
    return jobs


def bake_bank(
    profiles: list[str] | None = None,
    backend: str = "local",
    duration: float = 30.0,
    steps: int = 8,
) -> dict:
    targets = profiles or discover_profiles()
    if not targets:
        print("  no taste profiles found. Run `vt demo-profiles` first.")
        return {"baked": 0, "failed": 0}

    jobs = build_jobs(targets, duration, steps)
    BANK.mkdir(parents=True, exist_ok=True)

    print(f"  baking {len(jobs)} profile(s) via {acestep_dir()}")
    for j in jobs:
        print(f"    {j['id']}  seed={j['seed']}")
        print(f"      {j['caption'][:96]}")
    print()

    try:
        worker = AceStepWorker.get()
    except WorkerError as exc:
        print(f"  {exc}")
        print("  (or bake elsewhere and drop the .wav files into bank/)")
        return {"baked": 0, "failed": len(targets)}

    baked = failed = 0
    with stage("bake.all", profiles=len(jobs)) as m:
        for i, job in enumerate(jobs, 1):
            print(f"  [{i}/{len(jobs)}] {job['id']} ...", end="", flush=True)
            try:
                path = worker.generate(job)
            except WorkerError as exc:
                failed += 1
                print(f" FAILED: {exc}")
                continue
            profile, _hits, synth = _synth_for(job["id"], duration, steps)
            bank_add(
                job["id"], path, synth.params, backend,
                # Recorded so an unrehearsed live taste can match against it.
                centroid=taste_centroid(profile) if profile.positives else None,
            )
            baked += 1
            print(f" ok -> {path.name}")
        m.update({"baked": baked, "failed": failed})

    print(f"\n  done: {baked} baked, {failed} failed  ->  {BANK}")
    return {"baked": baked, "failed": failed}


def import_bank(src_dir: Path) -> int:
    """Adopt .wav files baked elsewhere (a GPU box, a colleague's machine).

    Filenames must be <profile_hash>.wav; the bank is keyed by hash so a mislabelled file
    would silently never be found.
    """
    from .prompt import load as load_prompt

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
