# Deck update — what changed in the numbers

Companion to `QDRANT-STORY.md`, which has already been updated in place. This is the
**diff**, so you can find and replace in the slides rather than re-read the whole thing.

Everything here was re-measured today against a live collection. Where a "before" number was
not measurable retrospectively, the original 1,005-track corpus was **re-ingested into a
parallel `music_small` collection** and the identical test run against both — so the two
columns are genuinely comparable, not one measured and one remembered.

**What happened:** the corpus grew from 1,005 to 8,780 tracks (every CC0/CC-BY track in FMA,
not just `fma_small`), and the generation bank was re-baked against it.

---

## 1. Collection — replace these outright

| metric | before | **after** |
|---|---|---|
| tracks | 1,005 | **8,780** |
| points (10s chunks) | 3,015 | **25,791** |
| segments (30s) | 1,005 | **8,828** |
| distinct artists | 202 | **1,594** |
| `audio` vectors | 3,015 | **25,791** |
| `text` vectors | 1,005 | **8,828** |
| total vectors | 4,020 | **34,619** |
| HNSW-indexed vectors | 0 | **30,772** |
| indexed payload fields | 6 | **7** |
| genres present | 8 | **14** |
| Qdrant memory | — | **136 MB** |

The 7th payload index is `is_upload`, from the bring-your-own-audio feature. **Still 512
dimensions, still cosine, still 2 named vectors, still one collection** — the architecture slides
are unaffected.

## 2. Latency — this one improves the story

Previously the deck said *"exact — at this size Qdrant builds no HNSW graph."* **That is now
false and must be replaced.** The graph exists; `exact=True` forces a full scan anyway,
because reproducibility matters more than speed for a rehearsed demo.

Median of 7, warm:

| | 3,015 points | **25,791 points** |
|---|---|---|
| Qdrant grouped query + payload | 10.2 ms | **24.7 ms** |
| Qdrant ungrouped query | 2.1 ms | **4.2 ms** |
| CLAP text embedding — *not Qdrant* | — | 10.9 ms |
| end to end | — | **35.6 ms** |

**Use 24.7 ms on the slide, not 59 ms.** The 59 ms figure I gave earlier bundled the
embedding and the HTTP stack. The clean statement: *8.6× the data, 2.4× the query time, on a
deliberately exact scan.*

## 3. Max-similarity example — swap the numbers

The old example track is no longer in the top results. Re-measured from a live query:

```
                        BEFORE                          AFTER
  Little Glass Men — Westside Chillers    Lee Rosevere — Going Home
  chunks 0.4274 · 0.3717 · 0.1831         chunks 0.4690 · 0.3969 · 0.3154
  max-sim     0.4274                      max-sim     0.4690
  mean-pooled 0.3274  (−0.10)             mean-pooled 0.3938  (−0.08)
```

The point is unchanged and still holds; only the track and the digits move.

## 4. Genre spread — the dashboard visualization slide

Both columns are **segment** counts now, so they are directly comparable. The old corpus was
8 deliberately *balanced* genres; the new one is the real, unbalanced distribution.

| genre | before | **after** |
|---|---|---|
| electronic | 648 | **1,868** |
| rock | 324 | **1,557** |
| old-time / historic | — | **1,345** |
| experimental | 411 | **990** |
| classical | — | **975** |
| hip-hop | 636 | **947** |
| pop | 168 | **754** |
| instrumental | 327 | **599** |
| folk | 342 | **532** |
| international | 159 | **207** |
| spoken · jazz · blues · easy listening | — | **78 · 72 · 30 · 21** |

**This strengthens the slide.** Before, an audience could fairly say the clusters separated
because the corpus was *built* balanced across 8 genres. Now the distribution is lopsided and
real — and the clusters still separate, from audio embeddings alone, with the tags applied
afterward as colour only.

## 5. The closing number

| | percentile |
|---|---|
| before — old corpus, old bank | 71.5 |
| new corpus, **stale** bank | 80.9 |
| **new corpus, re-baked bank** | **84.7** |

Say *"closer to your taste than 85% of 8,828 corpus segments."*

The middle row is worth knowing but not worth a slide: expanding changed what retrieval
returns, so the banked audio had been generated from prompts the corpus no longer produces.
Re-baking recovered ~4 points.

## 6. Retrieval quality — a new claim you can now make

Queries the old corpus could not serve now work. *"aggressive distorted metal riff"* returns
actual metal (Serpentarivm, Mouthus, Alpha Hydrae); 5 of the top 6 hits are tracks that did
not exist in the corpus before. With 202 artists you could not honestly demo genre reach.
With 1,594 you can.

---

## 7. Correction — one table in §4.3 overstates what negatives do

**This is the only thing here that weakens a claim, so it is the one to read.**

The deck's gesture table implies a negative reliably moves the list. Measured properly — 15
trials, 5 different queries, 3 different negatives each, identical on both collections:

| | negatives that visibly moved the top 12 |
|---|---|
| 1,005-track corpus | **4 / 15  (27%)** |
| 8,780-track corpus | **5 / 15  (33%)** |

**This is not a corpus-size effect** — it was already unreliable, and is marginally better
now. I raised it initially as a risk the expansion introduced; measuring both collections
showed that was wrong.

What it means for the talk: **an improvised negative on stage moves the list about a third of
the time.** The rehearsed demo profiles are not subject to this — `vt rehearse` passes
because their negatives were *chosen to contrast* — and it reports the real movement each
run (currently `1 in, 1 out, 4 moved`).

Two honest options for the slide:

- **Recommended:** demo the negative from a rehearsed profile, and state the gesture table as
  *"what a rehearsed contrast does"* rather than as a general property.
- Or keep it live and say out loud that a negative close to your positives may change little
  — which is itself a true and interesting fact about vector space, and safer than a moment
  that silently does nothing in front of an audience.

Re-measured per-gesture effect on a 12-row list, for reference:

| gesture | before | after |
|---|---|---|
| 1st positive | 9 new · 2 moved | 10 new · 1 moved |
| 2nd positive | 3 new · 8 moved | 9 new · 3 moved |
| a negative | 1 new · 2 moved | *varies — see above* |
| removing a mark | 4 new · 7 moved | 9 new · 3 moved |

---

## 8. Unchanged — no slide edits needed

- 512 dimensions, cosine, two named vectors, one collection.
- The five capabilities and their framing: cross-modal search, grouping/max-sim, the
  Recommendation API, the Discovery API, payload filtering.
- The Qdrant Cloud strict-mode teaching point (`unindexed_filtering_retrieve=false`).
- Everything in the honesty guardrails, except the `indexed_vectors_count: 0` bullet, which
  is now wrong — replace with: *don't turn `exact=True` into a performance slide; the demo
  forces a full scan for reproducibility, not because Qdrant needs one.*
- The corpus is still **CC0/CC-BY only**. It is 8,780 rather than FMA's 106,574 precisely
  because of that filter — which is a better version of the same point: the licence filter is
  the ceiling, not the infrastructure.
