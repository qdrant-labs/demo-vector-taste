# Vector Taste — the Qdrant story

Context for designing a talk. This describes **what the demo proves about vector search and
Qdrant**, and nothing about how the app is built. Every number here was measured against the
running collection, not estimated.

Talk: *Vector Taste: Using AI Composition to Co-Create Music* — Qdrant meetup, Austin, 35 min,
live on stage.

---

## 1. The one-sentence version

**You search a music library by *sound* instead of by tags, teach it your taste with
thumbs-up/thumbs-down on actual tracks, generate a new track from what it retrieves, and then
put that new track back into the same database to measure how close it landed.**

The whole arc is vector operations. There is no genre filter, no keyword index, no ML training
step, and no re-ranking service. Every stage is a query against one Qdrant collection.

---

## 2. Why this is a vector-search story and not a music story

Music is the ideal subject because **the thing you care about has no words for it.** Nobody
can type the query that finds "that warm, slightly sad, tape-saturated thing with the piano."
Tags can't express it — `hip-hop` is 324 of our tracks. Genre is a folder; taste is a
direction in space.

That is the general lesson, and it transfers to anything with the same shape: images, video,
audio logs, product photos, support calls, code. **When the useful similarity is perceptual
rather than lexical, the vector IS the query language.**

Three claims the demo makes concrete on stage:

| Claim | How it's shown |
|---|---|
| You can search a modality you can't describe | Type English, get audio back, with no text ever matched against text |
| You can steer results with examples instead of words | Two clicks (+/−) visibly reorder the list |
| You can measure how close generated content landed | Re-embed the output, rank it against the human corpus |

---

## 3. What's actually in Qdrant

One collection, `music_segments`.

```
3,015 points        one per 10-second audio window
1,005 segments      the 30-second clips a human actually listens to
512 dimensions      cosine distance
2 named vectors     audio + text, same space
6 payload indexes   segment_id, track_id, artist, tags, bpm, is_generated
```

**Named vectors** put both modalities in one collection rather than two databases:

- `audio` — on all 3,015 points. The sound itself.
- `text` — on 1,005 points, one per segment. A caption embedding.

Both are 512-d cosine, because they come from **CLAP**, a joint audio-text model whose two
towers share one space. (CLAP is an open model, not a Qdrant component — worth naming
honestly, and it means this pattern works with any joint embedder.)

**Why one point per 10 seconds, not one per track.** CLAP's audio tower has a hard 10-second
window; hand it 30 seconds and it takes a *random* 10-second crop, so the same file embeds
differently every run. Chunking explicitly makes the index reproducible. That creates a
problem — the user wants tracks, the index holds fragments — which Qdrant solves in the next
section.

---

## 4. The five Qdrant capabilities the demo leans on

### 4.1 Cross-modal search: text query, audio results

The query is a sentence. The search runs **against the `audio` vectors**, not the `text` ones.

That distinction is the entire point. Searching the text vectors would compare your words to
our captions — keyword search wearing a costume. Searching the audio vectors with a text
embedding retrieves *the sound itself*, and works on tracks that have no caption, no tags, and
no title worth reading.

> **Slide-worthy:** one collection, one query, two modalities, no join.

### 4.2 Grouping: max-similarity aggregation, server-side, free

`query_points_groups(group_by="segment_id", group_size=3)` rolls the 10-second chunks back up
to the 30-second segment **and scores each group by its best member.**

This is not a convenience — it changes the results. A real group from a live query:

```
Little Glass Men — Westside Chillers
  chunk scores:  0.4274   0.3717   0.1831
  max-sim:       0.4274   ← what Qdrant returns
  mean-pooled:   0.3274   ← what you'd get averaging, a 0.10 penalty
```

A track whose chorus is a perfect match and whose intro is silence should rank on the chorus.
Averaging buries the hook. **Qdrant does the max-sim rollup in the database**; the application
never re-ranks, never fetches all chunks, never post-processes.

> **Slide-worthy:** the fragment/document mismatch is universal — pages in a PDF, frames in a
> video, paragraphs in a contract. Grouping is the answer, and it costs one parameter.

### 4.3 The Recommendation API: taste as positive and negative examples

Marking tracks builds a query out of **point IDs, not vectors and not text**:

```
positive: [the tracks you liked]
negative: [the tracks you rejected]
strategy: best_score
```

Nothing is trained. Nothing is stored server-side. The "model" of your taste is a handful of
IDs, and it updates instantly on every click.

Qdrant offers three strategies, and the demo compares them rather than assuming:
`average_vector` collapses your picks to their midpoint; `best_score` ranks by the strongest
individual match; `sum_scores` rewards agreement across picks. **`best_score` was chosen**
because averaging two genuinely different likes lands the query in the empty space between
them — a place no real music lives.

Measured effect of each gesture on a 12-row result list:

| gesture | rows that are new | rows that moved |
|---|---|---|
| 1st positive | 9 | 2 |
| 2nd positive | 3 | 8 |
| a negative | 1 | 2 |
| removing a mark | 4 | 7 |

> **Slide-worthy:** "more like this, less like that" is a *database primitive*, not an ML
> project. This is the moment the audience should feel — the list visibly rearranges under a
> single click.

### 4.4 The Discovery API: direction instead of proximity

The demo also exercises `DiscoverQuery`, which takes a target plus **(positive, negative)
context pairs**. Each pair *partitions* the space rather than pulling toward an average:
"in the direction of A rather than B, near target."

Recommendation asks *what is close to my examples*. Discovery asks *which way is the good
direction*. Different questions, same collection, no extra infrastructure.

### 4.5 Payload filtering, and the Cloud detail worth teaching

Every query carries a filter excluding generated tracks (`is_generated`), so machine output
never contaminates human search results. Licensing is enforced in the payload too — the corpus
is filtered to CC0 and CC-BY only.

**The teaching moment:** Qdrant Cloud enables strict mode by default with
`unindexed_filtering_retrieve=false`. Filtering or grouping on an un-indexed payload field
works perfectly on a laptop and **errors on Cloud**. Every field this demo filters or groups
on is explicitly indexed for exactly that reason.

> **Slide-worthy:** this is a real, specific, actionable thing an audience of engineers can
> take home — the kind of detail that makes a talk trustworthy.

---

## 5. Closing the loop — the part that makes it falsifiable

The generated track is embedded with the same CLAP model and **upserted into the same
collection** it was retrieved from. Then it's ranked against the human corpus.

The headline is a **percentile, not a cosine**: "closer to your taste than 84% of 1,005
segments." A cosine of 0.61 means nothing to an audience and looks identical whether the demo
worked or not. A percentile is legible and it can be *wrong*, which is what makes it worth
showing.

Three deliberate choices keep that number honest, and they are all Qdrant queries:

- **Population is segments, not chunks** — each scored by its best chunk, the same max-sim
  rule retrieval uses. Ranking the generated track's one good chunk against other tracks'
  filler would inflate it.
- **The population excludes generated points**, so the comparison is against human music and
  doesn't drift as more tracks accumulate.
- **The baseline is the closest human track you did *not* pick.** Comparing against your own
  picks compares the answer to the question: a track you marked scores ~0.91 against your own
  taste centroid by construction.

The taste centroid itself is the mean of your positive vectors, re-normalized. Negatives
deliberately do **not** move it — they shape retrieval, but the centroid is "where the user
pointed," and subtracting negatives pushes it into empty space.

> **Slide-worthy:** the database is not just the retrieval layer, it's the **evaluation
> harness**. The same index that found the neighbors scores the result.

---

## 6. The dashboard visualization

Qdrant's Distance Matrix API computes distances **server-side** (`points/search/matrix`) so
raw vectors never cross the wire; the browser only does the 2-D layout. Colored by genre tag,
the clusters separate cleanly:

```
hip-hop 324 · electronic 312 · experimental 199 · rock 182
folk 172 · instrumental 154 · pop 88 · international 69
```

**Read the picture carefully, because it's the strongest slide in the deck:** those clusters
were formed *entirely from audio embeddings*. The genre tags are applied afterward, as color
only — they never touch retrieval. Genre falling out on its own is visual proof that the
vectors encode what the music actually sounds like.

---

## 7. Numbers worth putting on a slide

| | |
|---|---|
| Points / segments | 3,015 / 1,005 |
| Vector dimensions | 512, cosine |
| Named vectors per collection | 2 (`audio`, `text`) |
| Indexed payload fields | 6 |
| Chunk window | 10s (model limit), rolled up to 30s segments |
| Search mode | exact — at this size Qdrant builds no HNSW graph, so scores are reproducible |
| Rows that move on one +/− click | 2–8 of 12 |
| Max-sim vs mean-pooled, real group | 0.4274 vs 0.3274 |
| Generated track, typical result | ~84th percentile of 1,005 human segments |
| Closest human track (the baseline) | ~0.85 cosine |

---

## 8. Honesty guardrails — please don't overstate these

The talk's credibility rests on these being stated plainly. A designer should not "improve"
them into stronger claims:

- **The generated track does not beat human music.** It lands in the same neighborhood. The
  honest framing is "as close to your taste as most of the library," never "better."
- **The percentile is a similarity rank, not a quality score.** It says where the output sits
  relative to a taste centroid. It says nothing about whether the music is good.
- **CLAP is an open third-party model, not a Qdrant feature.** Qdrant stores, indexes,
  filters, groups, recommends, and scores. It does not embed. The pattern works with any
  joint embedder, which is a strength worth saying out loud.
- **The corpus is small on purpose** — 1,005 segments, CC0/CC-BY only, so every track shown on
  stage is legally clean. This is a demo of a *pattern*, not a benchmark.
- **`indexed_vectors_count: 0` in the dashboard is expected**, not a fault: below the indexing
  threshold Qdrant searches exactly. Don't let it become a slide about performance.

---

## 9. Vocabulary, in the order an audience meets it

| Term | Say it like this |
|---|---|
| Embedding | A list of 512 numbers describing what something sounds like |
| Joint / shared space | Text and audio described in the same numbers, so they can be compared |
| Cosine similarity | How close two of those descriptions point in the same direction |
| Point | One 10-second window of audio, plus its metadata |
| Payload | The metadata riding along with a vector — artist, BPM, license |
| Grouping | Rolling fragments back up into the thing a person cares about |
| Max-similarity | Score a track by its best moment, not its average |
| Recommendation API | Search using examples instead of a query |
| Discovery API | Search using a direction instead of a destination |
| Taste centroid | The middle of everything you said yes to |

---

## 10. The takeaway to land

**One collection did all of it:** cross-modal search, example-based recommendation,
directional discovery, metadata filtering, fragment-to-document grouping, and the final
measurement. No second store for the other modality. No re-ranking service. No training run.

The audience should leave thinking: *the interesting part of my data probably has no words for
it either — and the query language for that is a vector.*
