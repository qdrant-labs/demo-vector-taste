# Milestone 0 findings

What was verified against live sources, and what turned out to be wrong. Recorded because
most of these cost real debugging time and none of them are obvious from the docs.

Verified 2026-08-17 against qdrant-client 1.19.0, transformers 5.15.0, ACE-Step 1.5, and a
live Qdrant Cloud cluster.

---

## 1. `qdrant-client` removed three methods that every tutorial still uses

`client.search()`, `client.recommend()`, and `client.discover()` were **removed** in
**1.16.0** (PR #1103) — removed, not deprecated. Verified by grepping the tagged source:
three hits in v1.12–v1.15, **zero** in v1.16–v1.19.

| Old | Current |
|---|---|
| `client.search(...)` | `query_points(query=vec, using="audio")` |
| `client.recommend(...)` | `query_points(query=models.RecommendQuery(...))` |
| `client.discover(...)` | `query_points(query=models.DiscoverQuery(...))` |
| `client.search_batch(...)` | `query_batch_points(...)` |

`client.add()` / `client.query()` (the FastEmbed convenience wrappers) also went in 1.19.0.

**Consequence:** any Qdrant example older than roughly November 2025 will not run. Pin
`qdrant-client>=1.19,<2`.

## 2. CLAP's audio window is 10 seconds, not 30

Every `laion/larger_clap_*` checkpoint is *unfused*: `nb_max_samples: 480000` at 48 kHz,
with `truncation: rand_trunc`.

Feed it a 30-second segment and it embeds a **random 10-second crop** — the same file
produces a different vector on every call. That is silent: nothing errors, results just
become irreproducible, and any "distance from the taste centroid" measurement built on top
of it is noise.

**What we do instead:** one point per explicit 10s chunk, rolled back up to 30s segments at
query time with `query_points_groups(group_by="segment_id")`.

That turned out to be better than the obvious alternative anyway. Mean-pooling three chunks
into one segment vector buries the best moment — a segment whose chorus scores 0.8 and
whose intro scores 0.0 reports **~0.46 pooled but 0.8 under max-similarity**. Qdrant scores
a group by its best member, so grouping gives max-sim aggregation for free.

## 3. `transformers` v5 changed CLAP's return type

- `get_audio_features()` / `get_text_features()` now return `BaseModelOutputWithPooling`;
  the vector is in `.pooler_output`. In v4 they returned the tensor directly.
- `ClapProcessor` lost its custom `__call__`, so the v4 kwarg `audios=` is now `audio=`.
- The published Hugging Face docs still show the v4 form.

Outputs **are** already L2-normalized inside both getters in v4 and v5 — normalizing again
is a silent no-op at best. `embed.py` shims both with `getattr(out, "pooler_output", out)`.

## 4. ACE-Step: the version numbers in circulation are wrong

The brief specified "ACE-Step 1.5 Turbo (3.5B, Apache 2.0)". Every part of that is off:

| Claimed | Actual |
|---|---|
| 3.5B | That's ACE-Step **v1** (Apr 2025), the previous generation |
| Apache-2.0 | ACE-Step **1.5 is MIT** (equally permissive, but a different licence) |
| "1.5 Turbo" | Turbo checkpoints are **2B**; XL is larger still |

`acestep-v15-xl-turbo-diffusers` is an ~11GB repo — not viable in 16GB of unified memory.
We use the **2B turbo**.

## 5. ACE-Step 1.5 accepts BPM and key, and does not want tag strings

The brief said "ACE-Step does not accept key or scale, so key is filter-only" and asked for
a comma-separated tag string. Both are **v1 conventions**. In 1.5:

- `prompt` is a **free-form description**, not tags
- `bpm` (int), `keyscale` ("C major"), `timesignature` ("4") are first-class arguments
- `lyrics=""` for instrumentals
- `guidance_scale` is **ignored** on turbo checkpoints (they're guidance-distilled)
- seeding is `generator=`, not `seed=`; there is no `scheduler` argument

Audio style reference is real: `reference_audio` + `task_type="cover"` +
`audio_cover_strength`. The clip is internally normalized to 30s as 3x10s windows
(front/middle/back), so a 30s reference is exactly the right size.

## 6. ACE-Step and CLAP cannot share a virtualenv

ACE-Step 1.5 pins `transformers>=4.51,<4.58`. CLAP here needs `transformers>=5`. There is no
overlap.

ACE-Step therefore installs separately (`scripts/acestep_setup.sh`) and is reached over its
own localhost REST API. Still fully offline, and the model stays resident between requests —
which matters, because loading it costs far more than generating with it.

## 7. Qdrant Cloud rejects filtering on unindexed fields — confirmed on a live cluster

This is the one that would have broken the demo on stage.

Cloud enables strict mode **by default**:

```
strict_mode_config: enabled=True unindexed_filtering_retrieve=False
                    unindexed_filtering_update=False max_payload_index_count=100
```

Probed against a real cluster, **both** of these return **400 Bad Request** without a
payload index:

```
filter on is_generated  -> 400: Index required but not found for "is_generated"
                                of one of the following types: [bool]
group_by segment_id     -> 400: Index required but not found for "segment_id".
                                Help: Create an index supporting `match` for this key
```

The second one is the dangerous half: `group_by` is not a filter, so it is easy to assume it
needs no index. Both work fine against local Qdrant, so this only appears when you point at
Cloud. `store.INDEXES` indexes every field we filter or group on.

## 8. At demo scale, Qdrant builds no HNSW index — and that is good

Defaults are `indexing_threshold = 10000` KB and `full_scan_threshold = 10000` KB. A 512-d
float32 vector is 2 KiB, so HNSW starts at roughly **5000 vectors per segment**. With ~3000
chunk points spread over `num_cpus` segments we are an order of magnitude below it.

Confirmed on our own collection: `indexed_vectors: 0`, `status: green`.

So every query is already brute-force exact, and `search_params=SearchParams(hnsw_ef=128)`
does nothing at all. We pass `exact=True` instead — it costs nothing here, states the
intent, and keeps behaviour identical if the corpus later crosses the threshold.

It also means results are deterministic, which the demo depends on.

## 9. Equal scores have no tie-break

`ScoredPoint`'s `Ord` compares **score only** — no point-id tie-break. Equal-scored points
order by segment layout, which changes across restarts and re-ingests. Unlikely with CLAP
embeddings, but duplicate or silent chunks make it possible.

Both result levels are sorted client-side (`(-score, id)`), and `PointGroup` has no `.score`
attribute at all — the group's score is `hits[0].score`.

## 10. FMA's archives will not open with macOS `unzip`

They are bzip2-compressed (`compress_type 12`). The system `unzip` refuses with
`need PK compat. v4.6 (can do v4.5)` and **skips the files while exiting 0** — so it looks
like it worked. Python's `zipfile` handles them.

## 11. Licence distribution in FMA

Of 106,574 tracks, **8,780 are CC0 / Public Domain / plain CC-BY**. In the `fma_small`
subset we use: **1,005 of 8,000**.

The rest carry NonCommercial, ShareAlike, or NoDerivatives terms — all three disqualifying
here (ND makes a style-referenced generation legally awkward, SA propagates onto output, NC
blocks a company-published repo). Licence strings are free text with dozens of spellings, so
`corpus.is_permissive()` is deny-list-first and rejects anything unrecognized.

## 12. ACE-Step on an M4 Air: measured

No published Apple Silicon benchmark for ACE-Step 1.5 exists, so here is one.

**M4 MacBook Air, 16GB unified memory, 2B turbo DiT + 0.6B LM, MLX backend, 8 steps,
30-second instrumental:**

| Component | Time (s) | % of wall |
|---|---|---|
| Diffusion (8 steps) | 69.5 | 36.7% |
| VAE decode | 33.0 | 17.5% |
| Model offload | 49.6 | 26.2% |
| **DiT pipeline total** | **102.6** | **54.2%** |
| Overhead (I/O, audio save) | 86.6 | 45.8% |
| **TOTAL WALL TIME** | **189.2** | 100% |

**~3.2 minutes for 30 seconds of audio.** Live generation on stage is not viable — the
threshold for that was ~45s.

The per-step breakdown is the interesting part:

```
step 1: 48.72 s/it     <- MLX graph compilation
step 2: 14 s
steps 3-8: ~1-2 s/it
```

So the cost is almost entirely **one-time**: graph compilation plus model offload. A warm,
resident server amortizes both, which is why the `local` backend talks to a long-running
ACE-Step API rather than spawning a process per request. It also means **baking the bank
locally is practical** — no rented GPU required.

Two caveats on this measurement: corpus ingest was running concurrently, so it is a
pessimistic figure; and the first attempt auto-selected the **1.7B** language model rather
than the 0.6B the docs recommend for Apple Silicon, which thrashed swap for 13 minutes
without producing anything. Force `acestep-5Hz-lm-0.6B` on a 16GB machine.

## 13. The recommended CLAP checkpoint is degenerate on this corpus

The most consequential finding of the build, and it reverses a published benchmark.

`laion/larger_clap_music` is the obvious choice for a music demo: it is the music-only
checkpoint, and LAION's own table gives it **GTZAN 71%** zero-shot versus 51% for the
music+speech variant. We shipped it, ingested 1,005 tracks with it, and every text query
returned **the same top three results in the same order** — "aggressive loud distorted
metal" and "solo classical piano" were indistinguishable.

Measured, zero-shot text-to-audio genre classification on 32 tracks from this corpus:

| Checkpoint | Accuracy | Chance | Mean pairwise audio cosine |
|---|---|---|---|
| **`laion/larger_clap_general`** | **0.75** | 0.25 | **0.305** |
| `laion/clap-htsat-unfused` | 0.41 | 0.25 | 0.548 |
| `laion/larger_clap_music` | **0.25** | 0.25 | **0.922** |

**The music checkpoint sits exactly at chance.** The diagnostic number is the right-hand
column: mean pairwise cosine between *different* tracks is **0.92**, and two unrelated songs
embed at **0.9869**. The encoder has collapsed — it emits nearly the same vector for
everything, so there is no ranking signal for any query to exploit.

Things that were ruled out before blaming the checkpoint:

- **Not a loading bug.** `output_loading_info` reports no missing, unexpected, or
  mismatched keys.
- **Not the audio pipeline.** Files load at sensible RMS (~0.17), chunk into 3x10s as
  expected, and embedding is bit-identical across runs (cosine 1.000000).
- **Not normalization.** All vectors are unit length; text and audio both.
- **Not fixable by centering.** The audio mean vector has norm **0.96** (i.e. the whole
  corpus points one direction). Mean-centering triples the score spread, from 0.010 to
  0.037, but accuracy stays at chance — it redistributes noise rather than recovering
  signal.

**Lesson:** a published benchmark on someone else's dataset did not survive contact with
ours, and the failure was silent — plausible-looking ranked lists with plausible-looking
scores, all meaningless. `scripts/eval_embedders.py` makes this a one-command check, and
`mean pairwise cosine > 0.8` is the tripwire: it catches encoder collapse without needing
labels at all.

Worth noting the original research flagged this exact risk in advance — the HF checkpoint's
provenance is undocumented, and the advice was to benchmark both on our own data before
committing. That advice was correct and the answer was the opposite of the recommendation.

## Still open

- **Whether TwelveLabs Marengo's text tower is aligned to its audio tower.** Bedrock's own
  migration example passes `embeddingOption: "visual"` for text input.

## A note on CLAP score magnitudes

Cosine similarities from CLAP are compressed — real text-to-audio matches score around
**0.02-0.05**, not 0.6. Ranking is meaningful; the absolute number is not.

This is why the closing metric is a **percentile against the corpus** rather than a raw
cosine. "Cosine 0.024" tells an audience nothing and looks identical whether the demo
worked or not. "Closer to your taste than 94% of the library" is legible and falsifiable.
