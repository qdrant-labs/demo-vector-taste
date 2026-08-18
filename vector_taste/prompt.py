"""Prompt synthesis: retrieved payload + a user steer -> ACE-Step 1.5 parameters.

Deterministic template, no LLM. That is the design, not a limitation: the offline path is
the DEFAULT rather than a cached fallback, the same taste always yields the same prompt (so
a pre-baked bank entry is guaranteed to match), and there is no API key or network call
anywhere near the demo's critical path.

What changed, and why
---------------------
Prompts used to be built from FMA's `genre_top` tag, and the corpus has **9 distinct tags
across 1,006 segments**. Every taste collapsed to the same handful of phrases — three of the
four originally baked prompts contained "hip hop with a heavy beat, electronic production".

Now they are built from CLAP-derived descriptors (see `describe.py`), which are 98% unique
per segment, and **negatives actively shape the result**: a descriptor that is strong in the
positives *and* the negatives is not what distinguishes this taste, so it is dropped.
ACE-Step has no negative-prompt field, so this is the only way negatives can reach
generation at all — previously they only affected retrieval.

ACE-Step 1.5's actual signature, which differs from v1 and from most blog posts:
  - `prompt` (`caption` in the native API) is a FREE-FORM DESCRIPTION, not a tag string
  - `bpm` (int), `keyscale` ("C major"), `timesignature` ("4") are first-class arguments
  - `lyrics=""` for instrumentals (the native CLI uses "[Instrumental]")
  - `guidance_scale` is IGNORED on turbo checkpoints (they are guidance-distilled)
  - seeding is `generator=`, not `seed=`; there is no `scheduler` argument
  - `caption` is capped at 512 characters
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field

from .config import DATA
from .describe import FLAT
from .search import Hit

MAX_CAPTION = 512

# Genre tag -> descriptive language. Still useful as the backbone of the sentence; the CLAP
# descriptors supply everything around it.
_GENRE_PHRASING = {
    "hip-hop": "hip hop", "rock": "rock", "electronic": "electronic",
    "folk": "folk", "jazz": "jazz", "classical": "classical", "pop": "pop",
    "experimental": "experimental", "instrumental": "instrumental",
    "international": "world", "blues": "blues", "country": "country",
    "soul-rnb": "soul and rhythm and blues", "punk": "punk", "metal": "metal",
    "old-time": "old-time", "spoken": "spoken word", "easy listening": "easy listening",
}

_CATEGORY_OF = {term: cat for cat, term in FLAT}

# Order the caption reads in. Mood first is deliberate: it is the part a listener notices
# and the part a user's steer usually describes.
_ORDER = ["mood", "instrument", "production", "texture"]


SEED_MAX = 2**31 - 1


def seed_from_hash(profile_hash: str) -> int:
    """A stable seed derived from the taste profile.

    Used where reproducibility is the point: `vt bake` (the same profile must re-bake to the
    same track) and `vt rehearse` (which asserts the finale number does not drift).

    Every bank entry previously used seed=42, so identical noise plus near-identical prompts
    produced near-identical audio.
    """
    return int(profile_hash[:8], 16) % SEED_MAX


def fresh_seed() -> int:
    """A new random seed, so composing twice gives two different tracks.

    This is the default for live generation. The prompt is deterministic — the same taste
    describes the same music — but the seed is not, which is what makes each compose a new
    performance of that description rather than a replay of the same one.
    """
    import secrets

    return secrets.randbelow(SEED_MAX)


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
    task_type: str = "text2music"
    audio_cover_strength: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Synthesis:
    params: GenerationParams
    evidence: dict = field(default_factory=dict)


def _mode_bpm(hits: list[Hit]) -> int | None:
    """Median BPM of the retrieved set.

    Median rather than mean: one half-time or double-time detection error from librosa
    (a common failure — 70 vs 140) would drag a mean badly off.
    """
    vals = sorted(h.payload.get("bpm") for h in hits if h.payload.get("bpm"))
    return int(vals[len(vals) // 2]) if vals else None


def _dominant_key(hits: list[Hit]) -> str | None:
    """Most common key in the retrieved set, as a major scale.

    Mode, not average — keys are categorical. Reported as major because a chroma-based
    estimate does not reliably distinguish relative major/minor, and a wrong mode is
    audibly worse than an unspecified one.
    """
    keys = [h.payload.get("key") for h in hits if h.payload.get("key")]
    return f"{Counter(keys).most_common(1)[0][0]} major" if keys else None


def _tags(hits: list[Hit], top: int = 2) -> list[str]:
    c: Counter[str] = Counter()
    for h in hits:
        for t in h.payload.get("tags") or []:
            c[t.lower()] += 1
    return [t for t, _ in c.most_common(top)]


def _descriptor_scores(hits: list[Hit]) -> Counter[str]:
    """Frequency of each descriptor across a set of hits, normalized by set size.

    Weighted by position: `describe.top_descriptors` emits each category's best term first,
    so an earlier term is a stronger claim about the audio.
    """
    c: Counter[str] = Counter()
    if not hits:
        return c
    for h in hits:
        descs = h.payload.get("descriptors") or []
        for i, d in enumerate(descs):
            c[d] += 1.0 / (1 + i * 0.15)
    for k in c:
        c[k] /= len(hits)
    return c


def contrastive_descriptors(
    positives: list[Hit], negatives: list[Hit] | None = None, per_category: int = 1
) -> dict[str, list[str]]:
    """Descriptors that characterise the positives *and not* the negatives.

    A term common to both is not what the user is selecting for, so subtracting the negative
    side is what makes a rejection audible in the generated track rather than only in the
    ranking.
    """
    pos = _descriptor_scores(positives)
    neg = _descriptor_scores(negatives or [])

    net = {t: pos[t] - neg.get(t, 0.0) for t in pos}
    out: dict[str, list[str]] = {}
    for cat in _ORDER:
        terms = [(t, s) for t, s in net.items() if _CATEGORY_OF.get(t) == cat and s > 0]
        terms.sort(key=lambda kv: -kv[1])
        out[cat] = [t for t, _ in terms[:per_category]]
    return out


def synthesize(
    hits: list[Hit],
    steer: str = "",
    duration: float = 30.0,
    negatives: list[Hit] | None = None,
    seed: int | None = None,
    steps: int = 8,
    vocals: bool = False,
) -> Synthesis:
    """Build ACE-Step parameters from the retrieved neighborhood plus a user steer.

    The user's steer leads: they are steering, and burying their words behind aggregated
    corpus descriptors would invert the co-creation claim this demo makes.
    """
    tags = _tags(hits)
    bpm = _mode_bpm(hits)
    keyscale = _dominant_key(hits)
    desc = contrastive_descriptors(hits, negatives)

    genre = " ".join(_GENRE_PHRASING.get(t, t) for t in tags[:2]).strip()

    parts: list[str] = []
    if steer.strip():
        parts.append(steer.strip().rstrip("."))

    mood = desc.get("mood") or []
    instruments = desc.get("instrument") or []
    production = desc.get("production") or []
    texture = desc.get("texture") or []

    core = " ".join(x for x in [", ".join(mood), genre] if x).strip()
    if core:
        parts.append(core if not parts else f"in the style of {core}")
    if instruments:
        parts.append(" and ".join(instruments))
    parts.extend(production)
    parts.extend(texture)
    if bpm:
        parts.append(f"around {bpm} BPM")

    if not parts:
        parts = ["vocal music" if vocals else "instrumental music"]

    # "instrumental, no vocals" only once — the old template appended it while `instrumental`
    # was also a corpus tag, so some prompts said it twice. Skipped entirely when the user
    # asked for vocals, or the prompt would contradict the request.
    prompt = ", ".join(p for p in parts if p)
    if not vocals and "no vocals" not in prompt:
        prompt += ", instrumental, no vocals"

    params = GenerationParams(
        prompt=prompt[:MAX_CAPTION],
        lyrics="",  # instrumental for v1; lyrics need a lyrics-aware retrieval path
        audio_duration=float(duration),
        bpm=bpm,
        keyscale=keyscale,
        num_inference_steps=int(steps),
        seed=int(seed) if seed is not None else 42,
    )
    return Synthesis(
        params=params,
        evidence={
            "n_hits": len(hits),
            "n_negatives": len(negatives or []),
            "tags": tags,
            "descriptors": desc,
            "steer": steer,
            "vocals": vocals,
            "top_neighbors": [h.label for h in hits[:3]],
        },
    )


def cache_path(profile_hash: str):
    return DATA / f"prompt_{profile_hash}.json"


def save(profile_hash: str, synth: Synthesis):
    """Cache to disk so the bank and the live path provably use identical parameters."""
    DATA.mkdir(parents=True, exist_ok=True)
    p = cache_path(profile_hash)
    p.write_text(
        json.dumps({"params": synth.params.to_dict(), "evidence": synth.evidence}, indent=2)
    )
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
        f"  keyscale      {p.keyscale or '(unspecified)'}",
        f"  duration      {p.audio_duration:.0f}s   steps {p.num_inference_steps}",
        f"  seed          {p.seed}",
        f"  lyrics        {'(instrumental)' if not p.lyrics else p.lyrics[:40]}",
        "",
        f"  from          {s.evidence.get('n_hits', 0)} neighbors"
        f", {s.evidence.get('n_negatives', 0)} negative(s)",
    ]
    for cat in _ORDER:
        terms = (s.evidence.get("descriptors") or {}).get(cat) or []
        if terms:
            lines.append(f"    {cat:<11s} {', '.join(terms)}")
    for n in s.evidence.get("top_neighbors", []):
        lines.append(f"                - {n[:56]}")
    return "\n".join(lines)
