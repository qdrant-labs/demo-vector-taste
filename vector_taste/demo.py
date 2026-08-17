"""Build the scripted demo taste profiles.

The bank is keyed by taste-profile hash, so baking needs profiles to exist first. These are
generated deterministically from the scripted queries rather than hand-picked, so a rebuilt
corpus regenerates the same demo path instead of leaving dangling point IDs in the bank.

Each profile takes the top hit as a positive and a deliberately *distant* hit as a negative
— the last result of an opposing query. A negative drawn from the same result list barely
moves anything; the contrast has to be real for the on-stage diff to be visible.
"""

from __future__ import annotations

from .search import by_text
from .taste import TasteProfile, diff, recommend

# (name, query, contrasting query used to source the negative)
DEMO_SPECS = [
    ("lofi", "dreamy lo-fi hip hop with vinyl crackle and mellow keys", "aggressive loud distorted metal"),
    ("electronic", "driving electronic beat with heavy bass and synths", "quiet solo acoustic guitar"),
    ("acoustic", "warm acoustic guitar, intimate and sparse", "loud electronic dance music"),
]


def build_demo_profiles(verbose: bool = True) -> list[TasteProfile]:
    profiles = []
    for name, query, contrast in DEMO_SPECS:
        hits = by_text(query, limit=8)
        if not hits:
            if verbose:
                print(f"  {name}: no results for {query!r}, skipped")
            continue

        opposing = by_text(contrast, limit=8)
        # Exclude anything that also matched the positive query — a negative that is already
        # a positive tells the recommender nothing.
        pos_ids = {h.segment_id for h in hits}
        negatives = [h for h in opposing if h.segment_id not in pos_ids]

        profile = TasteProfile(
            positives=[h.point_id for h in hits[:2]],
            negatives=[negatives[0].point_id] if negatives else [],
            steer=query,
        )
        profile.save(name)
        profile.save()  # also under its hash, which is what the bank looks up
        profiles.append(profile)

        if verbose:
            print(f"  {name:12s} {profile.hash}  +{len(profile.positives)} -{len(profile.negatives)}")
            print(f"               top: {hits[0].label[:52]}")
            if negatives:
                print(f"               neg: {negatives[0].label[:52]}")
            if profile.negatives:
                base = TasteProfile(positives=profile.positives, steer=profile.steer)
                d = diff(recommend(base, limit=10), recommend(profile, limit=10))
                verdict = (
                    f"{len(d.added)} in, {len(d.dropped)} out, {len(d.moved)} moved"
                    if d.changed
                    else "NO VISIBLE CHANGE — pick a stronger contrast"
                )
                print(f"               negative effect: {verdict}")
    return profiles
