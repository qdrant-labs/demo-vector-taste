"""Build the scripted demo taste profiles.

The bank is keyed by taste-profile hash, so baking needs profiles to exist first. These are
generated deterministically from the scripted queries rather than hand-picked, so a rebuilt
corpus regenerates the same demo path instead of leaving dangling point IDs in the bank.

The negative is drawn from INSIDE the current result set, not from an opposing query. That
is not a detail — it is the whole reason the demo works.

A maximally-contrasting negative ("metal" against a lo-fi search) sits far outside the
top-10, so it prunes nothing and the list does not move. Measured on this corpus:

    negative from an opposing query   -> 0 in, 0 out, 0 moved   (nothing happens)
    negative at rank 3 of the results -> 5 in, 5 out, 3 moved   (half the list changes)

It is also the honest interaction. Nobody refines a search by naming music they were not
shown; they hear something they dislike *in the results* and reject it.
"""

from __future__ import annotations

from .search import by_text
from .taste import TasteProfile, diff, recommend

# (name, query, rank of the result to reject)
DEMO_SPECS = [
    ("lofi", "dreamy lo-fi hip hop with vinyl crackle and mellow keys", 2),
    ("electronic", "driving electronic beat with heavy bass and synths", 2),
    ("acoustic", "warm acoustic guitar, intimate and sparse", 2),
]


def build_demo_profiles(verbose: bool = True) -> list[TasteProfile]:
    profiles = []
    for name, query, neg_rank in DEMO_SPECS:
        hits = by_text(query, limit=12)
        if len(hits) < neg_rank + 1:
            if verbose:
                print(f"  {name}: too few results for {query!r}, skipped")
            continue

        positives = [h.point_id for h in hits[:2]]
        base = TasteProfile(positives=positives, steer=query)
        ranked = recommend(base, limit=10)

        # Reject something the user can actually see and hear in the list. Skip anything
        # already marked positive — rejecting your own example teaches nothing.
        pos_ids = set(positives)
        candidates = [h for h in ranked if h.point_id not in pos_ids]
        negative = candidates[min(neg_rank, len(candidates) - 1)] if candidates else None

        profile = TasteProfile(
            positives=positives,
            negatives=[negative.point_id] if negative else [],
            steer=query,
        )
        profile.save(name)
        profile.save()  # also under its hash, which is what the bank looks up
        profiles.append(profile)

        if verbose:
            print(f"  {name:12s} {profile.hash}  +{len(profile.positives)} -{len(profile.negatives)}")
            print(f"               top: {hits[0].label[:52]}")
            if negative:
                print(f"               rejected: {negative.label[:48]}")
                d = diff(ranked, recommend(profile, limit=10))
                verdict = (
                    f"{len(d.added)} in, {len(d.dropped)} out, {len(d.moved)} moved"
                    if d.changed
                    else "NO VISIBLE CHANGE — try a different rank"
                )
                print(f"               negative effect: {verdict}")
    return profiles
