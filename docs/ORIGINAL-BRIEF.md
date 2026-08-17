We're building the demo for a conference talk called "Vector Taste: Using AI Composition to Co-Create Music" (Qdrant meetup, Austin, 35 minutes, live on stage). Read this whole brief, then start with Milestone 0 only.

## What It Does

A music library you search by sound instead of tags. You seed it with audio or a description, refine with positive AND negative examples, and the retrieved neighbors then drive an open-source model that composes a new track. Finally we re-embed the generated track and measure how close it landed to the target region.

The talk's argument is co-creation: the human stakes out a position by ear, the machine executes it. The demo has to make that legible on a projector.

## Hard Constraints

These are demo-day requirements, not preferences. Design for them from the start.

1. **Zero network dependency at runtime.** Venue wifi will fail. Qdrant runs in local Docker with a persisted volume, all models are pre-downloaded to a local cache, and the prompt-synthesis LLM call must have a cached-response fallback so the demo survives with no internet.
2. **Every stage runs standalone from the CLI** before any UI exists. I need to be able to debug one stage on stage without the app in the way.
3. **Log wall-clock timing for every stage** into a timings file. Stage durations determine my running order, so I need real numbers, not estimates.
4. **Pre-baked outputs are first-class.** The generation bank ships in the repo, keyed by taste profile. Live generation is a bonus path that falls back to pre-baked on any failure, loudly in the logs and silently on screen.
5. This repo ships publicly after the talk. README with one-command setup, no secrets committed, requirements pinned.

## Stack

- Qdrant via Docker for storage and retrieval
- CLAP for the joint audio-text embedding space
- ACE-Step 1.5 Turbo (3.5B, Apache 2.0) for generation
- Python, uv or venv, your call
- Demo UI later: single page, dark, large type, keyboard-driven. Not a notebook.

## Data Model

Collection `music_segments`, one point per 30-second segment, named vectors:

- `audio`: CLAP audio tower output, cosine
- `text`: CLAP text tower output for the segment caption, same space, cosine
- `lyrics`: leave out of v1, note where it would go

Payload: `track_id`, `artist`, `title`, `segment_index`, `start_sec`, `end_sec`, `bpm`, `key`, `tags[]`, `caption`, `license`, `source_url`, `is_generated`, `generation_run_id`.

Index `bpm`, `tags`, `artist`, and `is_generated` for filtering.

## Milestones

Build in this order. Do not start a milestone until the previous one runs clean from the CLI and I have said go.

**0. Verify and scaffold.** Do not trust your training data on library versions or model IDs. Check current docs for: the `qdrant-client` Python API for named vectors, the Recommendation API with negative examples, and the Discovery API with context pairs; the correct CLAP model on Hugging Face and its embedding dimension; the ACE-Step 1.5 repo, its Turbo variant, and its actual input signature. Report what you found and what conflicts with what I wrote above. Then scaffold the repo, docker-compose for Qdrant, config module, and timing logger. Nothing else.

**1. Ingest.** Corpus loader, 30-second segmenter, CLAP embedding, Qdrant upsert with named vectors and payload indexes. Start with a small public-domain or Creative Commons sample set so we can iterate fast. Make the corpus source swappable, since the real one will include opt-in tracks from Austin artists.

**2. Search, three ways.** Text to audio, audio to audio, and both combined. CLI only. Print results as a readable table with scores.

**3. Taste refinement.** This is the centerpiece of the talk, so it gets the most care. Recommendation API with multiple positives and negatives, then Discovery API with context pairs. I need to be able to show the result set visibly change when a negative is added, so build a diff view that highlights what dropped out and what moved in.

**4. Prompt synthesis.** Take retrieved payload (captions, tags, BPM) plus a user text steer, and emit an ACE-Step tag string: comma-separated genre, mood, two or three instruments, vocal type, production style, BPM. Lyrics field stays empty for instrumentals. Note that ACE-Step does not accept key or scale, so key is filter-only. Cache every synthesized prompt to disk for the offline fallback.

**5. Generate.** ACE-Step 1.5 Turbo with two inputs: the tag string from milestone 4, and the top-ranked neighbor's audio segment as a style reference clip. Verify ACE-Step's supported reference clip length before wiring it. Build the pre-baked bank generator here too.

**6. Close the loop.** Embed the generated track with the same CLAP model, upsert with `is_generated=true`, and report cosine distance from the taste centroid built in milestone 3. This is how the talk ends, so the output needs to be a single clear number plus a simple visual.

**7. Demo UI.** Only after 1 through 6 work from the CLI. Ask me before you design it.

## Stop and Ask Me

- If CLAP and an alternative like MuQ-MuLan look meaningfully different in quality for this corpus, benchmark both and bring me the numbers rather than picking.
- If ACE-Step's real input signature differs from what I described.
- Before adding any dependency that needs a network call at runtime.
- Before writing the UI.

Start with Milestone 0. Tell me what you verified and what I got wrong.
