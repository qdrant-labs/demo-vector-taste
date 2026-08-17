"""Compare CLAP checkpoints on THIS corpus, zero-shot.

Run this before trusting any embedding model, including the one this repo defaults to.
It exists because the published benchmarks disagreed with reality here: `larger_clap_music`
has the better published GTZAN score but is degenerate on this data, emitting near-identical
vectors for completely different tracks.

Two numbers are reported per checkpoint:

  accuracy              zero-shot text->audio genre classification. Compare against chance.
  pairwise audio cosine mean similarity between DIFFERENT tracks. This is the tell: a
                        healthy encoder spreads them out (~0.3). Anything above ~0.8 means
                        the encoder has collapsed and retrieval cannot work, whatever its
                        published score says.

Usage:
    uv run python scripts/eval_embedders.py                     # default checkpoints
    uv run python scripts/eval_embedders.py laion/clap-htsat-fused
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import AutoProcessor, ClapModel

from vector_taste.config import COLLECTION, SAMPLE_RATE, get_client
from vector_taste.embed import chunk_audio, load_audio

DEFAULT_MODELS = [
    "laion/larger_clap_general",
    "laion/larger_clap_music",
    "laion/clap-htsat-unfused",
]
GENRES = ["hip-hop", "folk", "rock", "electronic", "classical", "jazz"]
PER_GENRE = 8


def sample_corpus(per_genre: int = PER_GENRE) -> dict[str, list[str]]:
    """One audio path per track, grouped by top genre tag."""
    pts, _ = get_client().scroll(
        COLLECTION, limit=8000, with_payload=["tags", "audio_path", "chunk_index"]
    )
    by: dict[str, list[str]] = {}
    for p in pts:
        pl = p.payload or {}
        if pl.get("chunk_index") != 0:
            continue
        tags = pl.get("tags") or [""]
        by.setdefault(tags[0], []).append(pl["audio_path"])
    return {g: by[g][:per_genre] for g in GENRES if len(by.get(g, [])) >= 4}


def evaluate(model_id: str, corpus: dict[str, list[str]]) -> dict:
    model = ClapModel.from_pretrained(model_id).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    def embed(paths):
        out = []
        for p in paths:
            chunk = chunk_audio(load_audio(p))[:1]  # first 10s window is enough here
            inputs = proc(audio=chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                o = model.get_audio_features(**inputs)
            out.append(getattr(o, "pooler_output", o).float().numpy()[0])
        return np.array(out)

    genres = list(corpus)
    audio = np.vstack([embed(corpus[g]) for g in genres])
    labels = sum([[g] * len(corpus[g]) for g in genres], [])

    ti = proc(text=[f"{g} music" for g in genres], return_tensors="pt", padding=True)
    with torch.inference_mode():
        to = model.get_text_features(**ti)
    text = getattr(to, "pooler_output", to).float().numpy()

    pred = [genres[i] for i in (audio @ text.T).argmax(1)]
    acc = float(np.mean([p == lab for p, lab in zip(pred, labels, strict=True)]))

    sim = audio @ audio.T
    iu = np.triu_indices(len(audio), 1)
    return {
        "accuracy": acc,
        "chance": 1 / len(genres),
        "pairwise": float(sim[iu].mean()),
        "n": len(audio),
    }


def main(argv: list[str]) -> int:
    models = argv[1:] or DEFAULT_MODELS
    corpus = sample_corpus()
    if not corpus:
        print("no corpus found — run `vt ingest` first")
        return 2

    print(f"\n  {sum(len(v) for v in corpus.values())} tracks across {len(corpus)} genres: "
          f"{', '.join(corpus)}\n")
    print(f"  {'checkpoint':40s} {'acc':>6} {'chance':>7} {'pair-cos':>9}  verdict")
    print("  " + "-" * 82)

    for m in models:
        try:
            r = evaluate(m, corpus)
        except Exception as exc:  # noqa: BLE001
            print(f"  {m:40s} FAILED: {str(exc)[:34]}")
            continue
        if r["pairwise"] > 0.8:
            verdict = "COLLAPSED — unusable"
        elif r["accuracy"] > r["chance"] * 2:
            verdict = "good"
        elif r["accuracy"] > r["chance"] * 1.2:
            verdict = "weak"
        else:
            verdict = "at chance"
        print(f"  {m:40s} {r['accuracy']:>6.2f} {r['chance']:>7.2f} "
              f"{r['pairwise']:>9.3f}  {verdict}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
