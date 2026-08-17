"""FastAPI backend. Imports the same modules the CLI uses — no duplicated logic.

Audio is streamed from disk rather than embedded, so the page stays small and the browser
can seek.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import BANK, GEN_BACKEND, ROOT, is_cloud
from ..search import Hit, by_text, format_table  # noqa: F401

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Vector Taste", docs_url=None, redoc_url=None)


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


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/api/status")
def status():
    from ..store import collection_info

    try:
        info = collection_info()
    except Exception as exc:
        raise HTTPException(503, f"Qdrant unavailable: {exc}") from exc
    return {
        "points": info["points"],
        "target": "cloud" if is_cloud() else "local",
        "backend": GEN_BACKEND,
    }


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

    after = recommend(profile, limit=req.limit)
    payload = {
        "hits": [_hit_json(h) for h in after],
        "profile": profile.to_dict(),
        "diff": None,
    }

    if req.negatives:
        before = recommend(TasteProfile(req.positives, [], req.steer), limit=req.limit)
        d = diff(before, after)
        payload["diff"] = {
            "dropped": [_hit_json(h) for h in d.dropped],
            "added": [h.segment_id for h in d.added],
            "moved": {h.segment_id: [old, new] for h, old, new in d.moved},
            "changed": d.changed,
        }
    profile.save()
    return payload


@app.post("/api/generate")
def generate_route(req: TasteReq):
    from ..generate import generate
    from ..prompt import save as save_prompt
    from ..prompt import synthesize
    from ..taste import TasteProfile, recommend, taste_centroid

    profile = TasteProfile(req.positives, req.negatives, req.steer)
    if profile.is_empty():
        raise HTTPException(400, "mark at least one positive first")

    hits = recommend(profile, limit=10)
    synth = synthesize(hits, steer=req.steer)
    save_prompt(profile.hash, synth)
    profile.save()

    # Pass the centroid so an unrehearsed taste falls back to the NEAREST banked track
    # rather than silence. Every live gesture produces a new hash.
    centroid = taste_centroid(profile) if profile.positives else None
    res = generate(synth.params, profile.hash, centroid=centroid)
    return {
        "profile": profile.hash,
        "prompt": synth.params.prompt,
        "bpm": synth.params.bpm,
        "keyscale": synth.params.keyscale,
        "backend": res.backend,
        "from_bank": res.from_bank,
        # Surfaced so a fallback is visible to the presenter without a stack trace on screen
        "note": res.note,
        "audio_url": f"/generated/{res.path.name}",
        "reference": _hit_json(hits[0]) if hits else None,
    }


@app.post("/api/loop")
def loop_route(req: TasteReq):
    from ..generate import bank_best_match
    from ..loop import close_loop
    from ..taste import TasteProfile, taste_centroid

    profile = TasteProfile(req.positives, req.negatives, req.steer)
    if not profile.positives:
        raise HTTPException(400, "mark at least one positive first")
    # Score the audio the user actually heard, including a nearest-match fallback.
    audio, _ = bank_best_match(profile.hash, taste_centroid(profile))
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
    path = (BANK / name).resolve()
    if not path.is_file() or BANK.resolve() not in path.parents:
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="audio/wav")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
