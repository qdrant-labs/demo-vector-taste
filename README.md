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

uv run vt corpus                      # what corpus is available, ingested, on disk
uv run vt upload my-track.mp3         # embed your own audio, print its neighbors
uv run vt reset --uploads             # drop every upload

uv run vt timings                     # wall-clock per stage
uv run vt preflight                   # pre-demo checklist
```

---

## Making the corpus bigger

The corpus is **1,005 tracks because that is every permissively-licensed track in
`fma_small`** — 1,005 of its 8,000 pass the CC0/CC-BY filter. The licence filter is the
ceiling, not an ingest limit. `vt corpus` prints where you stand without touching the network.

FMA's archives are nested, all 30-second clips:

| subset | tracks | usable (CC0/CC-BY) | whole archive | **selective fetch** |
|---|---|---|---|---|
| `small` | 8,000 | 1,005 | 7.7 GB | — |
| `medium` | 25,000 | 2,345 | 23.8 GB | ~2.3 GB |
| `large` | 106,574 | **8,780** | 100.3 GB | **~8.8 GB** |

```bash
uv run vt fetch --subset large        # ~7 GB, not 100 GB
uv run vt ingest --subset large
```

Measured on an M4 MacBook Air doing exactly that: **7,205 tracks fetched in 7.19 GB with zero
failures at 11.7 tracks/s**, then ingested at ~4 tracks/s into **25,791 points across 8,828
segments by 1,594 artists**. Search stays fast — an exact scan of the whole collection is
59 ms.

After expanding, **re-run `vt bake`**: retrieval returns different neighbours, so the banked
audio was generated from prompts the corpus no longer produces. Measured here — the finale
went 80.9 with the stale bank to 84.7 once it was re-baked against the larger corpus.

**`vt fetch` pulls only the tracks you can actually use.** Downloading `fma_large.zip` to keep
8,780 of its 106,574 files would discard 92% of a 100 GB transfer — and would not fit on a
laptop. Instead the ZIP's central directory is read over HTTP Range (4 requests, 8.7 MB), and
each wanted member is fetched with a single ranged request and checked against the CRC in that
directory. Verified byte-for-byte against tracks we already had from `fma_small`. Pass
`--full-archive` for the old download-and-extract path, and `--limit N` to take a slice.

Ingest is additive and point IDs are deterministic, so expanding is idempotent and **saved
taste profiles keep working**. Two things do change: the closing percentile is computed
against a bigger population, so that number moves; and retrieval returns new neighbours, so
generation prompts differ from the ones the bank was baked with. **Re-run `vt bake` before a
talk** if you expand.

---

## Bring your own music

Drop an audio file anywhere on the page (or press **Upload**). It is chunked into the same
10-second windows as the corpus, embedded with the same CLAP model, and upserted into the
same Qdrant collection — then **Find similar** returns the tracks nearest to it.

Because an upload is an ordinary point carrying `is_generated: false`, it needs no special
handling in retrieval: it is searchable, markable with +/−, and usable as a recommendation
example exactly like a corpus track. A 3½-minute file becomes ~7 segments / 21 chunk points
and appears as one clip; **Find similar** queries with all of them at once, so the match is
made on the strongest moment of the track rather than on its intro.

`.mp3 .wav .flac .m4a .ogg`, up to 30 MB, truncated at 5 minutes.

**Four things this deliberately does with somebody else's music:**

- **Uploads are cleared on every server start**, so a run always begins on the fixed corpus.
  `uploads/` is gitignored on top of the global `*.mp3`/`*.wav` rules.
- **The licence label reads `your upload`, never a CC string.** The result row renders that
  field, and printing `CC-BY` over a stranger's file would be a false claim in a repo whose
  argument is that its corpus is legally clean. `ATTRIBUTIONS.md` cannot pick uploads up.
- **Your audio is never sent to a hosted generator.** A hosted backend uploads its style
  reference to a third party, so on those backends the reference skips past uploads to the
  first corpus track. Local ACE-Step still conditions on your own clip — that never leaves
  the machine.
- **The closing percentile still counts the corpus, not your uploads.** Uploads are part of
  the searchable library, but letting them into the scoring population would make the
  headline number depend on whatever was dropped in and stop it being comparable run to run.

The filename you upload is display text only — files are stored under a generated name, so a
filename like `../../etc/passwd.mp3` is inert.

---

## Generating music

Generation has five backends, selected with `GEN_BACKEND` — or picked live from the
**Bank · Local · ElevenLabs** toggle beside **Compose**, which overrides the default for that
one track. The toggle starts on whatever `GEN_BACKEND` says, so the stage config still wins by
default. **Search works without any of them.**

| `GEN_BACKEND` | Behavior | Needs | Cost |
|---|---|---|---|
| `local` *(default)* | composes a **new track every time** | Apple Silicon (MLX) or an NVIDIA GPU, `./scripts/acestep_setup.sh` | free, ~2 min/track |
| `bank` | replays pre-generated tracks, instantly | nothing | free |
| `elevenlabs` | hosted, **seconds instead of minutes** | an [ElevenLabs](https://elevenlabs.io) key | ~$0.075 / 30s track |
| `modal` | your own GPU deployment | a [Modal](https://modal.com) account | $30/mo free credits |
| `replicate` | hosted | a [Replicate](https://replicate.com) token | ~$0.02/track |

**Each compose is a new performance.** The prompt is deterministic — the same taste describes the
same music — but the seed is random per run, so composing the same taste twice gives two different
takes of that description. Both are kept, in `generated/`, named by seed. Pass `--seed N` (or
`--reproducible`) to re-hear an exact take.

### Two display modes

The toggle in the header (or `M`) switches between **Desktop** and **Presentation**.

Desktop is the default and is what you want while working: dense rows, BPM · key · licence
on every result, the raw cosine, and the keyboard legend. Presentation is for the projector —
type scales up, rows get taller, and the working chrome disappears so the back row reads
titles rather than metadata. Everything that carries the argument stays in both: the
percentile, both bars, the prompt, and the NEW/UP/DOWN tags.

The choice is stored in `localStorage` and applied before first paint, so there is no reflow
on load. **Rehearse in presentation mode** — it is not the default, and the density you
practise in should be the density you present in.

Actions are icons rather than words (search, upload, transport, theme, mode, stop, remove).
Each keeps an `aria-label` and a hover tooltip, and the icons are an inline SVG sprite, so a
page load still makes zero external requests. `+` / `−` stay as symbols: they match the
on-screen copy and the keyboard shortcuts, and their shapes survive a washed-out projector.

**Use `GEN_BACKEND=bank` on stage.** A two-minute wait in front of an audience is not acceptable,
so the bank replays tracks pre-generated by `vt bake`. It is instant and works offline, at the cost
of the same taste always returning the same audio. Any backend falls back to the bank on failure —
loud in the logs, quiet on screen.

`modal` and `replicate` **cannot do audio style reference**; `local`, `modal`, and `elevenlabs` can
condition generation on a retrieved clip.

### ElevenLabs Music

The fast path — measured **3.7s** against ~120s locally, which is the difference between iterating
on a taste and waiting on one. Set `ELEVENLABS_API_KEY` (see `.env.example`) and the toggle enables
itself; without a key it stays disabled and explains why rather than failing on click.

Two things it does *better* than local, and three caveats worth knowing before you rely on it:

- **Negative examples reach the model.** Every other backend can only use a "less like this" mark to
  reshape retrieval. Here the rejected clip's descriptors go in as `negative_styles`, so the
  generator itself is told what to avoid. (Descriptors the rejected track *shares* with the accepted
  ones are filtered out first — otherwise the request would ask for and forbid the same thing.)
- **Style conditioning measurably works.** Same prompt, with and without the retrieved clip as a
  reference: 0.8686 vs 0.7202 cosine to that clip.
- **The seed is not reproducible.** Measured: the same seed produced different audio on consecutive
  calls, matching ElevenLabs' own "exact reproducibility is not guaranteed". Different seeds do vary
  the take. `vt bake` and `vt rehearse` therefore stay on `local`, where the seed genuinely is
  deterministic.
- **Instrumental is a strong hint, not a flag.** The API's `force_instrumental` cannot be combined
  with a composition plan, and a plan is required for both seed and conditioning — so instrumental is
  requested through the chunk text and `negative_styles`. It holds on measured output, but it is not
  a guarantee.
- **It needs the network**, so it is not the stage config. `vt preflight` flags that.

#### Vocals

The **Instrumental · Vocals** toggle beside Compose (`V`) is ElevenLabs-only, and disables
itself on the other generators rather than pretending. ACE-Step has no lyrics source, so
vocals there would be wordless syllables; the API refuses the combination with a 400 instead
of quietly handing back an instrumental.

There is a wrinkle worth knowing. In plan mode a chunk's `text` **is** the lyric content, and
this demo retrieves *sound* — there are no words to put there. So a vocal take makes an extra
call to `POST /v1/music/plan`, which writes a plan from our prompt, and we keep only its
lyrics: the styles, negatives, duration and audio conditioning are all still ours. Asking it
for lyrics explicitly matters — handed the bare sound description it planned `[Intro]` /
`[Groove]` with no words at all.

Measured on real output, CLAP-scored against "a song with singing voice and vocals" versus
"instrumental music, no vocals":

| | vocals | instrumental | verdict |
|---|---|---|---|
| toggle off | +0.2133 | **+0.2822** | instrumental |
| toggle on | **+0.2542** | +0.1144 | sings |

The extra call costs ~5s (8.5s total vs 3.4s instrumental). **Its price is unmeasured**: the
`character_count` on `/v1/user/subscription` does not move for music at all — not for the plan
call and not for a generation we know was billed — so that meter cannot answer the question.
It returns JSON rather than audio, and ElevenLabs bills music per minute of audio, but treat
that as reasoning rather than a measurement. If the lyric call fails the track still
generates, from a bare `[Verse]` marker.

**Do not commit ElevenLabs output.** Their Music terms prohibit, on the self-serve tiers, creating
"a library, catalog, database, or other repository of Output … making it available to third
parties" — which a public repo of generated tracks plausibly is. `generated/` is gitignored, and the
committed `bank/` is ACE-Step-only by design.

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
