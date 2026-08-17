"""Replay the full demo path end to end, with timings.

Two jobs. It produces the real per-stage wall-clock numbers that decide the talk's running
order, and it asserts the two properties that would embarrass you on stage:

  determinism — two runs must return byte-identical result ordering
  no drift    — the finale percentile must be identical on run 1, 2 and 3

The drift check is the important one. Generated points get upserted with is_generated=true;
if any query forgot to exclude them, the closing number would quietly change on the second
run. That is a bug you would only meet during the talk.
"""

from __future__ import annotations

import json

from .config import DATA
from .search import by_text
from .store import count, delete_generated
from .timing import stage

# The scripted path. Kept here so rehearsal and stage run exactly the same queries.
DEMO_QUERIES = [
    "dreamy lo-fi with vinyl crackle",
    "driving electronic beat with heavy bass",
    "warm acoustic guitar, intimate and sparse",
]


def _fingerprint(hits) -> str:
    return json.dumps([[h.segment_id, round(h.score, 6)] for h in hits])


def rehearse(reset: bool = True, runs: int = 3) -> bool:
    from .loop import close_loop
    from .generate import bank_lookup
    from .taste import TasteProfile, diff, recommend

    ok = True
    print()
    print("  REHEARSAL")
    print("  " + "-" * 76)

    if reset:
        n = delete_generated()
        print(f"  reset: purged {n} generated points")

    # --- search determinism -------------------------------------------------------
    prints = []
    for r in range(runs):
        run_fp = []
        for q in DEMO_QUERIES:
            with stage("rehearse.search", query=q):
                hits = by_text(q, limit=10)
            run_fp.append(_fingerprint(hits))
        prints.append(run_fp)

    if all(p == prints[0] for p in prints):
        print(f"  ok    search determinism   identical across {runs} runs")
    else:
        print(f"  FAIL  search determinism   results changed between runs")
        ok = False

    # --- taste refinement ---------------------------------------------------------
    profiles = sorted(DATA.glob("taste_*.json"))
    if not profiles:
        print("  skip  taste/loop           no saved taste profile")
        print("  " + "-" * 76)
        return ok

    profile = TasteProfile.load(profiles[0].stem.replace("taste_", ""))

    with stage("rehearse.taste"):
        after = recommend(profile, limit=10)
    if profile.negatives:
        base = TasteProfile(positives=profile.positives, steer=profile.steer)
        before = recommend(base, limit=10)
        d = diff(before, after)
        if d.changed:
            print(f"  ok    negative visibly moves the set "
                  f"({len(d.added)} in, {len(d.dropped)} out, {len(d.moved)} moved)")
        else:
            print("  FAIL  negative changed nothing — the centerpiece would fall flat")
            ok = False

    # --- finale stability ---------------------------------------------------------
    audio = bank_lookup(profile.hash)
    if not audio:
        print("  skip  finale               no banked audio (run `vt bake`)")
        print("  " + "-" * 76)
        return ok

    numbers = []
    for r in range(runs):
        with stage("rehearse.loop"):
            res = close_loop(audio, profile, upsert=True)
        numbers.append(round(res.percentile, 4))

    if len(set(numbers)) == 1:
        print(f"  ok    finale stability     {numbers[0]:.1f}th percentile on all {runs} runs")
    else:
        print(f"  FAIL  finale DRIFTED across runs: {numbers}")
        print("        generated points are polluting the population")
        ok = False

    print(f"  info  generated points left: {count(only_generated=True)} (purge with `vt reset`)")
    print("  " + "-" * 76)
    print(f"  {'READY' if ok else 'NOT READY'}\n")
    return ok
