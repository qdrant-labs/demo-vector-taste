"""Prompt synthesis: retrieved payload + a user steer -> ACE-Step 1.5 parameters.

Deterministic template, no LLM. That is the design, not a limitation: the offline path is
the DEFAULT rather than a cached fallback, the same taste profile always yields the same
prompt (so a pre-baked bank entry is guaranteed to match), and there is no API key or
network call anywhere near the demo's critical path.

ACE-Step 1.5's actual signature, which differs from v1 and from most blog posts:
  - `prompt` is a FREE-FORM DESCRIPTION, not a comma-separated tag string
  - `bpm` (int), `keyscale` ("C major"), `timesignature` ("4") are first-class arguments
  - `lyrics=""` for instrumentals (the native CLI uses "[Instrumental]")
  - `guidance_scale` is IGNORED on turbo checkpoints (they are guidance-distilled)
  - seeding is `generator=`, not `seed=`; there is no `scheduler` argument
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field

from .config import DATA
from .search import Hit

# Tag -> descriptive language. CLAP's captions and ACE-Step's prompts both respond better to
# natural phrasing than to bare genre labels.
_GENRE_PHRASING = {
    "hip-hop": "hip hop with a heavy beat",
    "rock": "guitar-driven rock",
    "electronic": "electronic production",
    "folk": "acoustic folk",
    "jazz": "jazz with live instrumentation",
    "classical": "orchestral classical",
    "pop": "polished pop",
    "experimental": "experimental textures",
    "instrumental": "instrumental",
    "international": "world music instrumentation",
    "blues": "blues phrasing",
    "country": "country instrumentation",
    "soul-rnb": "soulful rhythm and blues",
    "punk": "raw punk energy",
    "metal": "heavy distorted metal",
    "ambient": "ambient atmosphere",
    "trip-hop": "downtempo trip hop",
    "techno": "driving techno",
    "house": "house groove",
    "disco": "disco rhythm",
}


@dataclass
class GenerationParams:
    """Exactly the arguments ACE-Step 1.5 accepts. No invented fields."""

    prompt: str
    lyrics: str = ""  # empty = instrumental
    audio_duration: float = 30.0
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str = "4"
    num_inference_steps: int = 8  # turbo default
    shift: float = 3.0
    seed: int = 42
    task_type: str = "text2music"  # becomes "cover" when a reference clip is supplied
    audio_cover_strength: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Synthesis:
    params: GenerationParams
    evidence: dict = field(default_factory=dict)


def _mode_bpm(hits: list[Hit]) -> int | None:
    """Median BPM of the retrieved set.

    Median rather than mean: one half-time or double-time detection error from
    librosa (a common failure — 70 vs 140) would drag a mean badly off.
    """
    vals = sorted(h.payload.get("bpm") for h in hits if h.payload.get("bpm"))
    if not vals:
        return None
    return int(vals[len(vals) // 2])


def _dominant_key(hits: list[Hit]) -> str | None:
    """Most common key in the retrieved set, as a major scale.

    Mode, not average — keys are categorical. We report major because our chroma-based
    estimate does not distinguish relative major/minor reliably; a wrong mode would be
    audibly worse than an unspecified one.
    """
    keys = [h.payload.get("key") for h in hits if h.payload.get("key")]
    if not keys:
        return None
    return f"{Counter(keys).most_common(1)[0][0]} major"


def _tags(hits: list[Hit], top: int = 3) -> list[str]:
    c: Counter[str] = Counter()
    for h in hits:
        for t in h.payload.get("tags") or []:
            c[t.lower()] += 1
    return [t for t, _ in c.most_common(top)]


def synthesize(hits: list[Hit], steer: str = "", duration: float = 30.0) -> Synthesis:
    """Build ACE-Step parameters from the retrieved neighbourhood plus a user steer.

    The user's steer leads the prompt: they are steering, and burying their words behind
    aggregated corpus tags would invert the co-creation claim this demo is making.
    """
    tags = _tags(hits)
    bpm = _mode_bpm(hits)
    keyscale = _dominant_key(hits)

    described = [_GENRE_PHRASING.get(t, t) for t in tags]

    parts: list[str] = []
    if steer.strip():
        parts.append(steer.strip().rstrip("."))
    if described:
        parts.append(", ".join(described) if not parts else f"with {', '.join(described)}")
    if not parts:
        parts.append("instrumental music")

    prompt = ", ".join(parts)
    if bpm:
        prompt += f", around {bpm} BPM"
    prompt += ", instrumental, no vocals"

    params = GenerationParams(
        prompt=prompt,
        lyrics="",  # instrumental for v1; lyrics would need a lyrics-aware retrieval path
        audio_duration=float(duration),
        bpm=bpm,
        keyscale=keyscale,
    )
    return Synthesis(
        params=params,
        evidence={
            "n_hits": len(hits),
            "tags": tags,
            "bpm_values": [h.payload.get("bpm") for h in hits if h.payload.get("bpm")],
            "keys": [h.payload.get("key") for h in hits if h.payload.get("key")],
            "steer": steer,
            "top_neighbors": [h.label for h in hits[:3]],
        },
    )


def cache_path(profile_hash: str):
    return DATA / f"prompt_{profile_hash}.json"


def save(profile_hash: str, synth: Synthesis):
    """Cache to disk so the bank and the live path provably use identical parameters."""
    DATA.mkdir(parents=True, exist_ok=True)
    p = cache_path(profile_hash)
    p.write_text(json.dumps({"params": synth.params.to_dict(), "evidence": synth.evidence}, indent=2))
    return p


def load(profile_hash: str) -> Synthesis | None:
    p = cache_path(profile_hash)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return Synthesis(params=GenerationParams(**d["params"]), evidence=d.get("evidence", {}))


def format_synthesis(s: Synthesis) -> str:
    p = s.params
    lines = [
        "  prompt        " + p.prompt,
        f"  bpm           {p.bpm if p.bpm else '(unspecified)'}",
        f"  keyscale      {p.keyscale or '(unspecified)'}  [filter/generation metadata]",
        f"  duration      {p.audio_duration:.0f}s",
        f"  lyrics        {'(instrumental)' if not p.lyrics else p.lyrics[:40]}",
        f"  steps/shift   {p.num_inference_steps} / {p.shift}",
        "",
        f"  derived from  {s.evidence.get('n_hits', 0)} neighbors, tags={s.evidence.get('tags')}",
    ]
    for n in s.evidence.get("top_neighbors", []):
        lines.append(f"                - {n[:56]}")
    return "\n".join(lines)
