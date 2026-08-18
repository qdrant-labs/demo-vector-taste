"""CLI: every stage runs standalone.

Deliberate, from the brief: on stage you need to be able to debug one stage without the app
in the way. Uses argparse rather than a CLI framework — subcommands are the only feature
needed and stdlib means one less dependency to pin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import COLLECTION, DATA, GEN_BACKEND, is_cloud


def _p(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- fetch / ingest
def cmd_fetch(args):
    from .fetch import fetch_all

    fetch_all(audio=not args.metadata_only)
    return 0


def cmd_ingest(args):
    from .ingest import run

    res = run(limit=args.limit, subset=args.subset)
    _p(
        f"\ningested {res['tracks']} tracks -> {res['points']} chunk points"
        f"  (skipped {res['skipped_missing_audio']} without audio)"
    )
    _p("wrote ATTRIBUTIONS.md")
    return 0


def cmd_bootstrap(args):
    from .fetch import fetch_all
    from .ingest import run

    fetch_all(audio=True)
    res = run(limit=args.limit, subset="small")
    _p(f"\nready: {res['tracks']} tracks, {res['points']} points. Run `vt ui`.")
    return 0


# --------------------------------------------------------------------------------- search
def cmd_search(args):
    from .search import by_audio, by_text, combined, format_table

    if args.text and args.audio:
        hits = combined(args.text, args.audio, limit=args.limit, text_weight=args.text_weight)
        title = f'combined: "{args.text}" + {Path(args.audio).name} (w={args.text_weight})'
    elif args.text:
        hits = by_text(args.text, limit=args.limit)
        title = f'text -> audio: "{args.text}"'
    elif args.audio:
        hits = by_audio(args.audio, limit=args.limit)
        title = f"audio -> audio: {Path(args.audio).name}"
    else:
        _p("error: need --text and/or --audio")
        return 2

    _p("")
    _p(format_table(hits, title))
    if hits:
        _p("")
        _p("  point IDs (use as --pos / --neg for `vt taste`):")
        for i, h in enumerate(hits[:5], 1):
            _p(f"    {i}. {h.point_id}  {h.label[:44]}")
    return 0


# ---------------------------------------------------------------------------------- taste
def cmd_taste(args):
    from .search import format_table
    from .taste import TasteProfile, diff, format_diff, recommend

    profile = TasteProfile(positives=args.pos or [], negatives=args.neg or [], steer=args.steer or "")
    if profile.is_empty():
        _p("error: need at least one --pos")
        return 2

    if args.diff and profile.negatives:
        # The on-stage moment: same positives, with and without the negatives.
        base = TasteProfile(positives=profile.positives, steer=profile.steer)
        before = recommend(base, limit=args.limit)
        after = recommend(profile, limit=args.limit)
        _p("")
        _p(format_table(before, f"BEFORE  ({len(base.positives)} positive)"))
        _p("")
        _p(format_table(after, f"AFTER   (+ {len(profile.negatives)} negative)"))
        _p("")
        _p("  what changed:")
        _p(format_diff(diff(before, after)))
        hits = after
    else:
        hits = recommend(profile, limit=args.limit, strategy=args.strategy)
        _p("")
        _p(format_table(hits, f"taste: {len(profile.positives)}+ / {len(profile.negatives)}-"))

    path = profile.save()
    _p(f"\n  profile {profile.hash} saved -> {path.name}")
    return 0


def cmd_strategies(args):
    """Compare recommendation strategies side by side. Used to freeze one before rehearsal."""
    from qdrant_client import models

    from .search import format_table
    from .taste import TasteProfile, recommend

    profile = TasteProfile(positives=args.pos or [], negatives=args.neg or [])
    if profile.is_empty():
        _p("error: need at least one --pos")
        return 2
    for strat in (
        models.RecommendStrategy.AVERAGE_VECTOR,
        models.RecommendStrategy.BEST_SCORE,
        models.RecommendStrategy.SUM_SCORES,
    ):
        _p("")
        _p(format_table(recommend(profile, limit=args.limit, strategy=strat), f"strategy: {strat}"))
    return 0


# --------------------------------------------------------------------------------- prompt
def cmd_prompt(args):
    from .prompt import format_synthesis, save, seed_from_hash, synthesize
    from .taste import TasteProfile, negative_hits, recommend

    profile = TasteProfile.load(args.profile) if args.profile else TasteProfile(
        positives=args.pos or [], negatives=args.neg or [], steer=args.steer or ""
    )
    hits = recommend(profile, limit=args.limit)
    synth = synthesize(
        hits, steer=args.steer or profile.steer, duration=args.duration,
        negatives=negative_hits(profile), seed=seed_from_hash(profile.hash),
        steps=args.steps,
    )
    _p("")
    _p("ACE-Step parameters:")
    _p(format_synthesis(synth))
    _p(f"\n  cached -> {save(profile.hash, synth).name}")
    return 0


# ------------------------------------------------------------------------------- generate
def cmd_generate(args):
    from .generate import generate
    from .prompt import load as load_prompt
    from .prompt import save, synthesize
    from .taste import TasteProfile, negative_hits, recommend

    profile = TasteProfile.load(args.profile)
    hits = recommend(profile, limit=args.limit)
    # A cached prompt is reused only when pinning the seed. Otherwise every run must
    # re-synthesize so it picks up a NEW seed -- reusing the cache would replay the
    # previous take.
    # A cached prompt was built for one vocal setting; reusing it across the toggle would
    # silently generate the other one.
    synth = load_prompt(profile.hash) if (args.seed or args.reproducible) else None
    if synth is not None and bool(synth.evidence.get("vocals")) != bool(args.vocals):
        synth = None
    if synth is not None:
        synth.params.seed = _resolve_seed(args, profile.hash)
    if synth is None:
        synth = synthesize(
            hits, steer=profile.steer, duration=args.duration,
            negatives=negative_hits(profile),
            seed=_resolve_seed(args, profile.hash),
            steps=args.steps,
            vocals=args.vocals,
        )
        save(profile.hash, synth)

    # Reference clip: a 30s window centered on the WINNING chunk, not the raw segment.
    # Retrieval scored the segment by its best chunk, so handing the model the whole segment
    # could condition it on the intro that did not match.
    ref = None
    if hits and not args.no_reference:
        ref = _reference_clip(_reference_hit(hits, args.backend))

    from .taste import taste_centroid

    centroid = taste_centroid(profile) if profile.positives else None
    res = generate(
        synth.params, profile.hash, reference_audio=ref,
        backend=args.backend, centroid=centroid, vocals=args.vocals,
    )
    _p("")
    _p(f"  backend    {res.backend}{'  (from bank)' if res.from_bank else ''}")
    _p(f"  vocals     {'yes' if args.vocals else 'no (instrumental)'}")
    _p(f"  reference  {ref.name if ref else '(none)'}")
    _p(f"  audio      {res.path}")
    if res.note:
        _p(f"  note       {res.note}")
    return 0


def _resolve_seed(args, profile_hash: str) -> int:
    """--seed N pins exactly; --reproducible derives from the taste; otherwise fresh."""
    from .prompt import fresh_seed, seed_from_hash

    if getattr(args, "seed", None):
        return int(args.seed)
    if getattr(args, "reproducible", False):
        return seed_from_hash(profile_hash)
    return fresh_seed()


def _reference_hit(hits, backend: str | None):
    """Which hit conditions the generation.

    Normally the top one. But a hosted backend UPLOADS this clip to a third party, and an
    upload is somebody's own file of unknown provenance -- so on those backends we skip past
    uploads to the first corpus track. Local generation keeps the top hit whatever it is:
    conditioning on your own upload is a good result and never leaves this machine.
    """
    from .generate import AUDIO_LEAVES_MACHINE

    if (backend or GEN_BACKEND) not in AUDIO_LEAVES_MACHINE:
        return hits[0]
    for h in hits:
        if not h.payload.get("is_upload"):
            return h
    _p("  note       every hit is an upload; generating without a style reference")
    return None


def _reference_clip(hit) -> Path | None:
    """Extract a 30s window centered on the hit's winning chunk."""
    import numpy as np
    import soundfile as sf

    from .config import ROOT, SAMPLE_RATE
    from .embed import load_audio

    rel = hit.payload.get("audio_path")
    if not rel:
        return None
    src = ROOT / rel
    if not src.exists():
        return None

    wav = load_audio(src)
    center = int(hit.start_sec + 5) * SAMPLE_RATE  # middle of the 10s winning chunk
    half = 15 * SAMPLE_RATE
    start = max(0, center - half)
    clip = wav[start : start + 2 * half]
    if len(clip) < SAMPLE_RATE:
        return None

    out = DATA / f"ref_{hit.segment_id.replace(':', '_')}.wav"
    DATA.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.asarray(clip, dtype="float32"), SAMPLE_RATE)
    return out


def cmd_bake(args):
    from .bake import bake_bank, import_bank

    if args.import_from:
        return 0 if import_bank(Path(args.import_from)) else 1
    bake_bank(profiles=args.profile, backend=args.backend,
              duration=args.duration, steps=args.steps)
    return 0


# ----------------------------------------------------------------------------------- loop
def cmd_loop(args):
    from .generate import audio_for_profile
    from .loop import close_loop
    from .taste import TasteProfile, taste_centroid

    profile = TasteProfile.load(args.profile)
    audio = Path(args.audio) if args.audio else audio_for_profile(
        profile.hash, taste_centroid(profile) if profile.positives else None
    )[0]
    if not audio or not Path(audio).exists():
        _p(f"error: no audio for profile {profile.hash}. Run `vt generate` or pass --audio.")
        return 2

    res = close_loop(Path(audio), profile, upsert=not args.no_upsert)
    _p(res.summary())
    return 0


# ------------------------------------------------------------------------------ utilities
def cmd_describe(args):
    from .describe import TERMS, describe_collection

    _p(f"  scoring {len(TERMS)} descriptors against stored audio vectors")
    _p("  (no audio is re-read -- this is a payload update, not a re-ingest)")
    r = describe_collection(per_category=args.per_category)
    _p(f"\n  described {r['segments']} segments -> {r['points']} points")
    return 0


def cmd_demo_profiles(args):
    from .demo import build_demo_profiles

    _p("")
    profiles = build_demo_profiles()
    _p(f"\n  {len(profiles)} demo profile(s) saved. Next: uv run vt bake")
    return 0


def cmd_timings(args):
    from .timing import summary

    _p(summary())
    return 0


def cmd_info(args):
    from .store import collection_info, count

    _p(f"  collection   {COLLECTION}")
    _p(f"  target       {'Qdrant Cloud' if is_cloud() else 'local'}")
    try:
        info = collection_info()
        _p(f"  points       {info['points']}  (generated: {count(only_generated=True)})")
        _p(f"  indexed_vec  {info['indexed_vectors']}   <- 0 is expected: exact search at this scale")
        _p(f"  status       {info['status']}")
    except Exception as exc:  # noqa: BLE001
        _p(f"  error        {exc}")
        return 1
    return 0


def cmd_upload(args):
    """Embed a local audio file into the collection, same as dropping it on the UI."""
    from pathlib import Path as _Path

    from .uploads import UploadError, ingest, save

    src = _Path(args.path)
    if not src.is_file():
        _p(f"error: no such file {src}")
        return 2
    try:
        path, track_id = save(src.read_bytes(), src.name)
        clip = ingest(path, track_id, src.name)
    except UploadError as exc:
        _p(f"error: {exc}")
        return 2
    _p("")
    _p(f"  track      {clip['title']}")
    _p(f"  points     {clip['points']} across {clip['segments']} segment(s)")
    _p(f"  duration   {clip['seconds']}s{'  (truncated)' if clip['truncated'] else ''}")
    _p(f"  bpm / key  {clip['bpm'] or '-'} · {clip['key'] or '-'}")
    _p("")
    from .search import format_table
    from .taste import TasteProfile, recommend

    _p(format_table(recommend(TasteProfile(positives=clip["point_ids"]), limit=args.limit),
                    "nearest in the library:"))
    return 0


def cmd_reset(args):
    from .store import delete_generated, ensure_collection
    from .uploads import purge as purge_uploads

    if args.all:
        ensure_collection(recreate=True)
        _p("  collection recreated (empty)")
    elif args.uploads:
        _p(f"  purged {purge_uploads()} upload points")
    else:
        _p(f"  purged {delete_generated()} generated points")
    return 0


def cmd_preflight(args):
    from .preflight import run_preflight

    return 0 if run_preflight(check_audio=not args.no_audio) else 1


def cmd_ui(args):
    import uvicorn

    _p(f"  http://{args.host}:{args.port}   (backend: {GEN_BACKEND})")
    uvicorn.run("vector_taste.ui.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_rehearse(args):
    from .rehearse import rehearse

    return 0 if rehearse(reset=not args.no_reset) else 1


# ------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="vt", description="Vector Taste — search music by sound, refine by ear."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download corpus + metadata")
    f.add_argument("--metadata-only", action="store_true")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("ingest", help="segment, embed, upsert, write attributions")
    i.add_argument("--limit", type=int)
    i.add_argument("--subset", default="small")
    i.set_defaults(func=cmd_ingest)

    b = sub.add_parser("bootstrap", help="fetch + ingest in one step")
    b.add_argument("--limit", type=int)
    b.set_defaults(func=cmd_bootstrap)

    s = sub.add_parser("search", help="text->audio, audio->audio, or both")
    s.add_argument("--text")
    s.add_argument("--audio")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--text-weight", type=float, default=0.5)
    s.set_defaults(func=cmd_search)

    t = sub.add_parser("taste", help="refine with positives and negatives")
    t.add_argument("--pos", action="append")
    t.add_argument("--neg", action="append")
    t.add_argument("--steer", default="")
    t.add_argument("--limit", type=int, default=10)
    t.add_argument("--diff", action="store_true", help="show what the negatives changed")
    t.add_argument("--strategy", default="best_score")
    t.set_defaults(func=cmd_taste)

    st = sub.add_parser("strategies", help="compare recommendation strategies")
    st.add_argument("--pos", action="append")
    st.add_argument("--neg", action="append")
    st.add_argument("--limit", type=int, default=8)
    st.set_defaults(func=cmd_strategies)

    pr = sub.add_parser("prompt", help="synthesize ACE-Step parameters")
    pr.add_argument("--profile")
    pr.add_argument("--pos", action="append")
    pr.add_argument("--neg", action="append")
    pr.add_argument("--steer", default="")
    pr.add_argument("--limit", type=int, default=10)
    pr.add_argument("--duration", type=float, default=30.0)
    pr.add_argument("--steps", type=int, default=8)
    pr.add_argument("--seed", type=int)
    pr.add_argument("--reproducible", action="store_true")
    pr.set_defaults(func=cmd_prompt)

    g = sub.add_parser("generate", help="compose a track")
    g.add_argument("profile")
    g.add_argument("--backend", default=None)
    g.add_argument("--limit", type=int, default=10)
    g.add_argument("--duration", type=float, default=30.0)
    g.add_argument("--steps", type=int, default=8)
    g.add_argument("--seed", type=int, help="pin the seed to reproduce an exact take")
    g.add_argument("--reproducible", action="store_true",
                   help="derive the seed from the taste instead of composing anew")
    g.add_argument("--no-reference", action="store_true")
    g.add_argument("--vocals", action="store_true",
                   help="sing rather than play (elevenlabs only)")
    g.set_defaults(func=cmd_generate)

    bk = sub.add_parser("bake", help="pre-generate the bank")
    bk.add_argument("--profile", action="append")
    bk.add_argument("--backend", default="local")
    bk.add_argument("--duration", type=float, default=30.0)
    bk.add_argument("--steps", type=int, default=8)
    bk.add_argument(
        "--import-from",
        metavar="DIR",
        help="adopt <profile_hash>.wav files baked on another machine",
    )
    bk.set_defaults(func=cmd_bake)

    lp = sub.add_parser("loop", help="re-embed the generated track and score it")
    lp.add_argument("profile")
    lp.add_argument("--audio")
    lp.add_argument("--no-upsert", action="store_true")
    lp.set_defaults(func=cmd_loop)

    ds = sub.add_parser("describe", help="tag every segment with CLAP descriptors")
    ds.add_argument("--per-category", type=int, default=2)
    ds.set_defaults(func=cmd_describe)

    sub.add_parser(
        "demo-profiles", help="build the scripted demo taste profiles"
    ).set_defaults(func=cmd_demo_profiles)
    sub.add_parser("timings", help="per-stage wall clock").set_defaults(func=cmd_timings)
    sub.add_parser("info", help="collection status").set_defaults(func=cmd_info)

    up = sub.add_parser("upload", help="embed your own audio file and show its neighbors")
    up.add_argument("path")
    up.add_argument("--limit", type=int, default=10)
    up.set_defaults(func=cmd_upload)

    r = sub.add_parser("reset", help="purge generated points (or everything)")
    r.add_argument("--all", action="store_true")
    r.add_argument("--uploads", action="store_true", help="purge user uploads only")
    r.set_defaults(func=cmd_reset)

    pf = sub.add_parser("preflight", help="pre-demo checklist")
    pf.add_argument("--no-audio", action="store_true")
    pf.set_defaults(func=cmd_preflight)

    rh = sub.add_parser("rehearse", help="replay the full demo path with timings")
    rh.add_argument("--no-reset", action="store_true")
    rh.set_defaults(func=cmd_rehearse)

    u = sub.add_parser("ui", help="run the web UI")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8000)
    u.set_defaults(func=cmd_ui)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        # Stop the ACE-Step child too. It is a separate process holding ~4GB, and leaving
        # it resident after Ctrl+C would strand that memory.
        from .progress import abort_current

        abort_current()
        _p("\ninterrupted")
        return 130
    except FileNotFoundError as exc:
        _p(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
