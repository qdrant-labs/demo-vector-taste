"""FastAPI backend. Imports the same modules the CLI uses — no duplicated logic.

Audio is streamed from disk rather than embedded, so the page stays small and the browser
can seek.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import BANK, GEN_BACKEND, GENERATED, ROOT, is_cloud
from ..search import Hit, by_text, format_table  # noqa: F401

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Vector Taste", docs_url=None, redoc_url=None)

# Module-level so it is not a call in a default argument (ruff B008).
_UPLOAD_FILE = File(...)


def _hit_json(h: Hit) -> dict:
    p = h.payload
    return {
        "segment_id": h.segment_id,
        "point_id": h.point_id,
        "score": round(h.score, 4),
        "artist": p.get("artist", "?"),
        "title": p.get("title", "?"),
        "bpm": p.get("bpm"),
        "key": p.get("key"),
        "tags": p.get("tags") or [],
        "license": p.get("license", ""),
        # Drives the YOURS badge. Uploads are ordinary library members otherwise.
        "is_upload": bool(p.get("is_upload")),
        "start_sec": h.start_sec,
        "audio_url": f"/audio/{h.segment_id}",
    }


class SearchReq(BaseModel):
    text: str = ""
    limit: int = 12


class TasteReq(BaseModel):
    positives: list[str] = []
    negatives: list[str] = []
    steer: str = ""
    limit: int = 12
    # Speed/quality trade for live generation. 15s roughly halves the wait.
    duration: float = 30.0
    steps: int = 8
    # Which generator to use for THIS compose. None -> the GEN_BACKEND default, so the
    # toggle can change generator without restarting the server.
    backend: str | None = None
    # Sing rather than play. Only some backends can; the route rejects the rest outright.
    vocals: bool = False
    # Pin the seed to re-hear an exact take instead of composing a new one.
    reproducible: bool = False


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.on_event("startup")
def _startup() -> None:
    """Pre-load the generation model in the background.

    The worker takes ~150s to load 9GB of checkpoints and reports NO progress while doing
    it, so paying that during startup turns the first compose from ~250s (150 of them
    featureless) into ~99s with a real moving ring.

    Also clears any audio left over from a previous session, so every run starts on the
    fixed corpus rather than on whatever the last person dropped in.

    Skipped for GEN_BACKEND=bank: the stage config never generates, so holding ~4GB would
    be pure cost. Never blocks -- search must work while this runs.
    """
    from ..uploads import purge

    try:
        purge()
    except Exception as exc:  # noqa: BLE001 - a purge failure must not stop the server
        import logging

        logging.getLogger("vector_taste.ui").warning("upload purge failed: %s", exc)

    if GEN_BACKEND == "bank":
        return
    from ..worker import prewarm

    prewarm()


@app.get("/api/status")
def status():
    from ..store import collection_info
    from ..worker import worker_state

    try:
        info = collection_info()
    except Exception as exc:
        raise HTTPException(503, f"Qdrant unavailable: {exc}") from exc
    from ..generate import VOCALS_BACKENDS, available_backends

    return {
        "points": info["points"],
        "target": "cloud" if is_cloud() else "local",
        "backend": GEN_BACKEND,
        "worker": worker_state(),
        "backends": available_backends(),
        # So the UI can disable the vocals toggle rather than fail on Compose.
        "vocals_backends": sorted(VOCALS_BACKENDS),
    }


@app.post("/api/abort")
def abort():
    """Stop the in-flight generation.

    Dispatches through the abort registry rather than the ACE-Step worker, so it works on
    whichever backend is actually running: ACE-Step kills its process, ElevenLabs closes its
    HTTP client. Wiring this to the worker directly would make Stop silently do nothing on
    a hosted backend.
    """
    from ..progress import abort_current

    return {"aborted": abort_current()}


@app.get("/api/progress")
def progress():
    """Latest generation progress. Polled by the UI while a compose is in flight."""
    from ..progress import PROGRESS
    from ..worker import worker_state

    snap = PROGRESS.snapshot()
    snap["worker"] = worker_state()
    return snap


@app.post("/api/upload")
async def upload(file: UploadFile = _UPLOAD_FILE):
    """Embed a user's audio into the same collection the corpus lives in.

    Synchronous: a few seconds of CLAP, and the answer is worth waiting for. The file's own
    name never reaches the filesystem -- see uploads.save().
    """
    from ..uploads import UploadError, ingest, save

    data = await file.read()
    try:
        path, track_id = save(data, file.filename or "")
    except UploadError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        return ingest(path, track_id, Path(file.filename or "upload").name)
    except UploadError as exc:
        path.unlink(missing_ok=True)          # never leave audio we could not embed
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(500, f"could not embed that file: {exc}") from exc


@app.get("/api/uploads")
def uploads_list():
    from ..uploads import listing

    return {"uploads": listing()}


@app.delete("/api/uploads/{track_id}")
def uploads_delete(track_id: str):
    from ..uploads import delete

    return {"deleted": delete(track_id)}


@app.post("/api/search")
def search(req: SearchReq):
    if not req.text.strip():
        return {"hits": []}
    return {"hits": [_hit_json(h) for h in by_text(req.text, limit=req.limit)]}


@app.post("/api/taste")
def taste(req: TasteReq):
    """Recommend with positives/negatives, and report the diff against positives-only.

    The diff is computed server-side so the UI can highlight exactly what the negative did
    — that visible change is the whole argument of the talk.
    """
    from ..taste import TasteProfile, diff, recommend

    profile = TasteProfile(req.positives, req.negatives, req.steer)
    if profile.is_empty():
        return {"hits": [], "diff": None, "profile": None}

    # A negative with nothing to anchor it is not "less like this", it is "maximally unlike
    # this" -- the query goes to the far side of the space. Measured: zero of the twelve
    # results you were looking at survive, and the top scores are NEGATIVE (-0.48). It reads
    # as the app throwing your search away, so refuse it rather than serve the antipode.
    if req.negatives and not req.positives:
        raise HTTPException(400, "mark something you like first — a negative on its own "
                                 "jumps to unrelated music")

    after = recommend(profile, limit=req.limit)
    payload = {
        "hits": [_hit_json(h) for h in after],
        "profile": profile.to_dict(),
        "diff": None,
    }

    # Needs positives AND negatives. The baseline is the positives-only ranking, and with no
    # positives that profile is empty -- which recommend() rightly refuses, so marking "-"
    # before ever marking "+" used to 500. A negatives-only query still works and is still
    # useful; there is simply nothing to diff it against yet.
    if req.negatives and req.positives:
        before = recommend(TasteProfile(req.positives, [], req.steer), limit=req.limit)
        d = diff(before, after)
        payload["diff"] = {
            "dropped": [_hit_json(h) for h in d.dropped],
            "added": [h.segment_id for h in d.added],
            "moved": {h.segment_id: [old, new] for h, old, new in d.moved},
            "changed": d.changed,
        }
    # Deliberately NOT saved here. Every +/- gesture hits this route, and persisting each
    # one litters data/ with profiles that `vt bake` would then try to bake. Profiles are
    # saved on compose, where the prompt cache and bank actually need them.
    return payload


@app.post("/api/generate")
def generate_route(req: TasteReq):
    from ..generate import VOCALS_BACKENDS, generate
    from ..prompt import fresh_seed, seed_from_hash, synthesize
    from ..prompt import save as save_prompt
    from ..taste import TasteProfile, negative_hits, recommend, taste_centroid

    profile = TasteProfile(req.positives, req.negatives, req.steer)
    if profile.is_empty():
        raise HTTPException(400, "mark at least one positive first")

    backend = req.backend or GEN_BACKEND
    if req.vocals and backend not in VOCALS_BACKENDS:
        # Reject rather than quietly hand back an instrumental: the user asked for a voice
        # and would have no way to tell the request was dropped.
        raise HTTPException(
            400, f"the {backend} generator cannot sing — switch to "
                 f"{' or '.join(sorted(VOCALS_BACKENDS))}"
        )

    hits = recommend(profile, limit=10)
    synth = synthesize(
        hits, steer=req.steer, duration=req.duration,
        # Fresh seed every compose: the prompt is deterministic (the same taste
        # describes the same music) but the seed is not, so each press of Compose
        # is a NEW performance of that description rather than a replay.
        negatives=negative_hits(profile),
        seed=seed_from_hash(profile.hash) if req.reproducible else fresh_seed(),
        steps=req.steps,
        vocals=req.vocals,
    )
    save_prompt(profile.hash, synth)
    profile.save()

    # Pass the centroid so an unrehearsed taste falls back to the NEAREST banked track
    # rather than silence. Every live gesture produces a new hash.
    centroid = taste_centroid(profile) if profile.positives else None
    from ..generate import GenerationAborted

    # Negative descriptors reach the model directly on backends with a negative prompt
    # (ElevenLabs `negative_styles`); elsewhere they are ignored.
    neg_desc = [
        d for h in negative_hits(profile) for d in (h.payload.get("descriptors") or [])
    ]
    try:
        res = generate(
            synth.params, profile.hash, centroid=centroid,
            backend=backend, negatives=neg_desc or None, vocals=req.vocals,
        )
    except GenerationAborted:
        # 499 (client closed request) rather than 500: nothing failed, the user stopped it.
        raise HTTPException(499, "generation aborted") from None
    return {
        "profile": profile.hash,
        "prompt": synth.params.prompt,
        "bpm": synth.params.bpm,
        "keyscale": synth.params.keyscale,
        "backend": res.backend,
        # What was actually produced, not what was asked for: a bank fallback serves a
        # pre-baked instrumental, and labelling that "with vocals" in the UI would be a lie.
        "vocals": req.vocals and not res.from_bank,
        "from_bank": res.from_bank,
        # Surfaced so a fallback is visible to the presenter without a stack trace on screen
        "note": res.note,
        "audio_url": f"/generated/{res.path.name}",
        "reference": _hit_json(hits[0]) if hits else None,
    }


@app.post("/api/loop")
def loop_route(req: TasteReq):
    from ..generate import audio_for_profile
    from ..loop import close_loop
    from ..taste import TasteProfile, taste_centroid

    profile = TasteProfile(req.positives, req.negatives, req.steer)
    if not profile.positives:
        raise HTTPException(400, "mark at least one positive first")
    # Score the audio the user actually heard, including a nearest-match fallback.
    audio, _ = audio_for_profile(profile.hash, taste_centroid(profile))
    if not audio:
        raise HTTPException(404, "generate a track first")
    r = close_loop(audio, profile, upsert=True)
    return {
        "cosine": round(r.cosine, 4),
        "percentile": round(r.percentile, 1),
        "population": r.population,
        "baseline_cosine": round(r.baseline_cosine, 4),
        "baseline_percentile": round(r.baseline_percentile, 1),
        "beats_baseline": r.beats_baseline,
        # WHICH human track that baseline is, so the UI can play the comparison rather
        # than only print its cosine.
        "baseline": _hit_json(r.baseline_hit) if r.baseline_hit else None,
    }


@app.get("/audio/{segment_id:path}")
def audio(segment_id: str):
    """Stream a corpus segment. Path comes from the payload, never from user input."""
    from qdrant_client import models

    from ..config import COLLECTION, get_client

    res = get_client().scroll(
        COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="segment_id", match=models.MatchValue(value=segment_id)
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )[0]
    if not res:
        raise HTTPException(404, "segment not found")

    rel = (res[0].payload or {}).get("audio_path")
    if not rel:
        raise HTTPException(404, "no audio for segment")
    path = (ROOT / rel).resolve()
    # Defence in depth: the path came from our own payload, but never serve outside ROOT.
    if not path.is_file() or ROOT.resolve() not in path.parents:
        raise HTTPException(404, "audio file missing")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/generated/{name}")
def generated(name: str):
    """Serve a generated take, or a banked one. Both dirs, never outside them."""
    for root in (GENERATED, BANK):
        path = (root / name).resolve()
        if path.is_file() and root.resolve() in path.parents:
            # ElevenLabs returns mp3, ACE-Step wav -- serve each as what it actually is.
            media = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
            return FileResponse(path, media_type=media)
    raise HTTPException(404, "not found")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
