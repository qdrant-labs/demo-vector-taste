# Vector Taste

**Search a music library by sound instead of tags. Refine it by ear. Then let the result compose
something new.**

Seed the search with a text description or an audio clip, refine with positive *and* negative
examples, and the retrieved neighbors drive an open-source model that writes a new track. Finally the
generated track is re-embedded and measured against your taste — so you can see how close the machine
landed to where you pointed it.

Built for the talk *"Vector Taste: Using AI Composition to Co-Create Music"* (Qdrant meetup, Austin).

```
text or audio query  ──▶  Qdrant  ──▶  neighbors  ──▶  ACE-Step  ──▶  new track
                            ▲            │                               │
                            │       + / − by ear                         │
                            └────────  taste centroid  ◀── re-embed ──────┘
```

| Piece | What it does |
|---|---|
| [Qdrant](https://qdrant.tech) | Named vectors, grouped max-similarity search, recommend/discover with negatives |
| [LAION CLAP](https://huggingface.co/laion/larger_clap_music) | Joint text↔audio embedding space, 512-d, Apache-2.0 |
| [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) | Music generation with audio style reference, MIT |

Everything runs locally and offline. No API key is required for the core demo.

---

## Quickstart

**Requirements:** Python is handled by `uv`. You need `ffmpeg` and either `podman` or `docker`.

```bash
brew install ffmpeg uv podman     # macOS; podman needs `podman machine init && podman machine start`
git clone https://github.com/qdrant-labs/demo-vector-taste
cd demo-vector-taste
uv sync
./scripts/qdrant_up.sh            # auto-detects podman or docker
uv run vt bootstrap               # fetch corpus, embed, upsert, write ATTRIBUTIONS.md
uv run vt ui                      # http://localhost:8000
```

`bootstrap` is the slow one — it downloads the corpus and embeds it. Every step is idempotent, so
re-running it is safe.

### Using Docker instead of podman

`scripts/qdrant_up.sh` uses whichever of `podman` or `docker` it finds, so the command above works
either way. If you'd rather use Compose:

```bash
docker compose up -d
```

Both give you Qdrant on `localhost:6333` with a persistent volume, pinned to `v1.19.0`. Nothing else
in the stack can tell the difference.

### Using your own Qdrant Cloud cluster

```bash
cp .env.example .env
# set QDRANT_URL and QDRANT_API_KEY, then skip qdrant_up.sh entirely
```

The [free tier](https://qdrant.tech/pricing/) needs no credit card (1 GB RAM, 4 GB disk, 1 node).
Two things to know: free clusters are **suspended after 1 week idle and deleted after 4 weeks**, and
the first request after a resume can take a minute — neither is a bug in this demo.

---

## Try it from the CLI

Every stage runs standalone, so you can debug one piece without the app in the way.

```bash
uv run vt search --text "dreamy lo-fi with vinyl crackle"   # text  -> audio
uv run vt search --audio path/to/clip.mp3                   # audio -> audio
uv run vt search --text "warm analog pads" --audio clip.mp3 # both

# taste refinement: the centerpiece
uv run vt taste --pos <segment_id> --pos <segment_id> --neg <segment_id> --diff

uv run vt prompt --profile <hash>     # retrieved payload -> ACE-Step params
uv run vt generate --profile <hash>   # compose
uv run vt loop --profile <hash>       # re-embed and score against your taste

uv run vt timings                     # wall-clock per stage
uv run vt preflight                   # pre-demo checklist
```

---

## Generating music

Generation has four backends, selected with `GEN_BACKEND`. **Search works without any of them.**

| `GEN_BACKEND` | Needs | Cost |
|---|---|---|
| `bank` *(default)* | nothing — plays pre-generated tracks | free |
| `local` | Apple Silicon (MLX) or an NVIDIA GPU | free |
| `modal` | a [Modal](https://modal.com) account | $30/mo free credits |
| `replicate` | a [Replicate](https://replicate.com) token | ~$0.02/track |

`bank` exists because a conference demo shouldn't depend on a GPU finishing on time. Any backend
falls back to `bank` on failure. Note that the hosted backends **cannot do audio style reference** —
only `local` and `modal` support conditioning generation on a retrieved clip.

---

## Licensing, and why it constrains the corpus

The corpus is filtered to **CC0 and CC-BY only**. Not a nicety — three clauses would each break
this demo:

- **ND (NoDerivatives)** — using a clip as a generation style reference arguably creates a derivative
  work. Excluded.
- **SA (ShareAlike)** — would propagate share-alike obligations onto generated output. Excluded.
- **NC (NonCommercial)** — this repo is published by a company. Excluded.

That rules out some popular music datasets, including MTG-Jamendo (CC-BY-NC-SA metadata,
non-commercial research terms) and the Live Music Archive.

`ATTRIBUTIONS.md` is **generated at ingest** from the `license` and `source_url` fields on every
point, never hand-maintained. Corpus audio is downloaded by a script and never committed.

To use your own music, add a fetch function in `vector_taste/corpus.py` that yields tracks with
`license` and `source_url` populated. Everything downstream is source-agnostic.

### On the generation model

ACE-Step 1.5's model card states it was "trained on legally compliant datasets." Worth being precise:
that claim is **not corroborated in the paper**, which describes a 27M-sample corpus without any
licensing or provenance statement. We use ACE-Step because it's MIT-licensed, runs locally, and
supports audio style reference — but the *verifiable* licensing story here is about **our corpus**,
not the model's training data. If you need documented training provenance,
[Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2) (CC-BY-4.0 weights, explicitly
licensed stock music) is better evidenced.

---

## What it looks like working

Search returns different, genre-appropriate results per query:

```
$ uv run vt search --text "aggressive loud distorted metal guitar"
   1   0.3772   NanowaR Of Steel — Heavy Metal Kibbutz
$ uv run vt search --text "solo classical piano, gentle"
   1   0.3890   Jason Shaw — Timen Passing
```

Rejecting one result visibly reorders the rest:

```
$ uv run vt taste --pos <id> --pos <id> --neg <id> --diff
  [-] DROPPED           Uncle Milk — On two hours sleep
  [+] NEW       0.7088  Jason Shaw — Feels Good 2 B
  [~] up   #4->#3   0.7144  Uncle Milk — Ruggles
  (1 in, 1 out, 5 moved, 2 unchanged)
```

And the closing number:

```
$ uv run vt loop <profile>
  The generated track lands at the 98th percentile.
  Closer to your taste centroid than 98% of the 1005 segments in the library.

    generated   cosine +0.7306   percentile  97.6
    best human  cosine +0.8218   percentile  99.6   <- retrieval baseline
```

Interactive stages are sub-second (`search` 0.59s, `taste` 0.02s, `loop` 0.43s). Ingesting
1,005 tracks takes ~7 minutes; baking one track takes ~2 minutes. Both are one-time.

## Notes for anyone building on this

A few things that cost real debugging time and aren't obvious from the docs.
[`docs/FINDINGS.md`](docs/FINDINGS.md) has all sixteen with measurements; these are the ones
most likely to bite you:

1. **`qdrant-client` removed `.search()`, `.recommend()`, and `.discover()`** in 1.16.0 — they're
   gone, not deprecated. Everything is `query_points()` / `query_points_groups()`. Tutorials older
   than ~Nov 2025 will not run.
2. **CLAP's audio window is 10 seconds, not 30.** Feed it longer audio and it takes a *random* 10 s
   crop, so your embeddings aren't reproducible. We store one point per 10 s chunk and roll up to
   segments with `query_points_groups`, which also gives max-similarity for free.
3. **`transformers` v5 changed CLAP's return type.** `get_audio_features()` returns an object; the
   vector is `.pooler_output`. The v4 `audios=` kwarg is now `audio=`. Published docs still show the
   old form.
4. **Qdrant Cloud turns on strict mode by default**, which rejects filtering on *unindexed* payload
   fields. A filter that works against local Qdrant can fail against Cloud. Index every field you
   filter or group on.
5. **At this corpus size Qdrant doesn't build an HNSW index at all** (the threshold is ~5000 vectors
   per segment), so search is already exact and `hnsw_ef` does nothing. We pass `exact=True` to say
   so out loud.
6. **The music-specific CLAP checkpoint is degenerate on this corpus.** `larger_clap_music` has the
   better published GTZAN score (71% vs 51%) and scores *exactly at chance* here, emitting cosine
   0.99 between unrelated songs. We use `larger_clap_general`. Run
   `uv run python scripts/eval_embedders.py` before trusting any embedder — the tell needs no
   labels at all: **mean pairwise cosine above ~0.8 means the encoder has collapsed.**
7. **Style conditioning is `text2music` + `reference_audio`, not `task_type="cover"`.** `cover`
   means re-record *this specific song* and requires `src_audio`; passing only `reference_audio`
   fails with "Task 'cover' requires source audio".
8. **ACE-Step and CLAP cannot share a virtualenv** — ACE-Step pins `transformers<4.58`, CLAP needs
   `>=5`. `scripts/acestep_setup.sh` installs it separately.

---

## License

Code: Apache-2.0. Corpus audio: CC0/CC-BY per track, see `ATTRIBUTIONS.md`.
Models are licensed by their authors — CLAP is Apache-2.0, ACE-Step 1.5 is MIT.
