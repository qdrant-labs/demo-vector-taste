"""Pre-generate the bank.

The bank is the stage path: generation is instant and cannot fail in front of an audience.
Run this offline, on whatever hardware you have, then commit or release the result.

Bake on a real GPU if the local benchmark is slow — the bank is hardware-independent once
the audio exists, so where it was produced does not affect the demo.
"""

from __future__ import annotations

from pathlib import Path

from .config import DATA
from .generate import bank_add, generate
from .prompt import load as load_prompt
from .prompt import save as save_prompt
from .prompt import synthesize
from .taste import TasteProfile, recommend
from .timing import stage


def discover_profiles() -> list[str]:
    """Every saved taste profile is a bank candidate."""
    return sorted(p.stem.replace("taste_", "") for p in DATA.glob("taste_*.json"))


def bake_one(profile_hash: str, backend: str = "local", duration: float = 30.0) -> Path | None:
    from .cli import _reference_clip

    profile = TasteProfile.load(profile_hash)
    hits = recommend(profile, limit=10)

    synth = load_prompt(profile.hash) or synthesize(hits, steer=profile.steer, duration=duration)
    save_prompt(profile.hash, synth)

    ref = _reference_clip(hits[0]) if hits else None

    with stage("bake.generate", profile=profile.hash, backend=backend) as m:
        res = generate(synth.params, profile.hash, reference_audio=ref, backend=backend)
        m["from_bank"] = res.from_bank
        m["note"] = res.note

    if res.from_bank and res.note:
        print(f"  {profile_hash}: FAILED — {res.note}")
        return None

    bank_add(profile.hash, res.path, synth.params, backend)
    print(f"  {profile_hash}: baked -> {res.path.name}")
    return res.path


def bake_bank(
    profiles: list[str] | None = None, backend: str = "local", duration: float = 30.0
) -> dict:
    targets = profiles or discover_profiles()
    if not targets:
        print("  no taste profiles found. Create one with `vt taste --pos ...` first.")
        return {"baked": 0, "failed": 0}

    print(f"  baking {len(targets)} profile(s) with backend={backend}")
    baked = failed = 0
    for h in targets:
        try:
            failed += bake_one(h, backend=backend, duration=duration) is None
            baked += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {h}: ERROR {exc}")
            failed += 1
    baked -= failed
    print(f"\n  done: {baked} baked, {failed} failed")
    return {"baked": baked, "failed": failed}
