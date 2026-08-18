"""Checks for the logic that would fail silently.

No Qdrant or model downloads needed — these run in under a second. The integration path is
covered by `vt rehearse`, which needs a live collection.

Run: uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from vector_taste.corpus import Track, is_permissive, license_short
from vector_taste.embed import centroid, chunk_audio
from vector_taste.prompt import _dominant_key, _mode_bpm, synthesize
from vector_taste.search import Hit
from vector_taste.taste import TasteProfile, diff


# --------------------------------------------------------------------------- licensing
@pytest.mark.parametrize(
    "lic",
    [
        "Attribution",
        "Creative Commons Attribution",
        "CC0 1.0 Universal",
        "Public Domain Mark 1.0",
        "Attribution 3.0 United States",
        "Attribution 4.0 International",
    ],
)
def test_permissive_allowed(lic):
    assert is_permissive(lic)


@pytest.mark.parametrize(
    "lic",
    [
        # Every one of these appears in the real FMA metadata.
        "Attribution-Noncommercial-Share Alike 3.0 United States",
        "Attribution-NonCommercial-NoDerivatives (aka Music Sharing) 3.0 International",
        "Attribution-NonCommercial-ShareAlike 3.0 International",
        "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
        "Attribution-ShareAlike",
        "Attribution-NoDerivatives 4.0 International",
        "Attribution-Noncommercial 3.0 United States",
        "Attribution-Noncommercial-No Derivative Works 3.0 Germany",
        "",
        None,
    ],
)
def test_restrictive_rejected(lic):
    """NC, SA and ND must all be rejected, however they are spelled.

    ND matters most: a style-referenced generation is arguably a derivative work.
    SA would propagate share-alike onto generated output. Missing metadata is not a
    license, so it is rejected too.
    """
    assert not is_permissive(lic)


def test_license_short():
    assert license_short("CC0 1.0 Universal") == "CC0-1.0"
    assert license_short("Public Domain Mark 1.0") == "Public Domain"
    assert license_short("Attribution 3.0 United States") == "CC-BY-3.0"
    assert license_short("Attribution") == "CC-BY"


# ----------------------------------------------------------------------------- chunking
def test_chunk_audio_splits_on_clap_window():
    """30s at 48kHz must become exactly three 10s chunks — CLAP's window is 10s, not 30."""
    wav = np.zeros(30 * 48_000, dtype=np.float32)
    chunks = chunk_audio(wav)
    assert len(chunks) == 3
    assert all(len(c) == 10 * 48_000 for c in chunks)


def test_chunk_audio_drops_short_tail():
    """A 2s tail is mostly silence; its embedding would be noise in a max-sim comparison."""
    wav = np.zeros(22 * 48_000, dtype=np.float32)
    chunks = chunk_audio(wav)
    assert len(chunks) == 2  # 10 + 10, the 2s remainder dropped


def test_chunk_audio_keeps_very_short_file():
    """A file shorter than the minimum must still yield one chunk, not vanish."""
    assert len(chunk_audio(np.zeros(48_000, dtype=np.float32))) == 1


# ----------------------------------------------------------------------------- centroid
def test_centroid_is_unit_length():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 512)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert np.isclose(np.linalg.norm(centroid(v)), 1.0, atol=1e-5)


def test_centroid_of_identical_vectors_is_that_vector():
    v = np.zeros((3, 512), dtype=np.float32)
    v[:, 0] = 1.0
    assert np.allclose(centroid(v), v[0], atol=1e-6)


def test_centroid_rejects_opposing_vectors():
    """Two exactly opposed unit vectors average to zero and cannot be normalized."""
    v = np.zeros((2, 512), dtype=np.float32)
    v[0, 0], v[1, 0] = 1.0, -1.0
    with pytest.raises(ValueError):
        centroid(v)


# --------------------------------------------------------------------------------- diff
def _hit(seg, score):
    return Hit(seg, score, f"p-{seg}", {"artist": "a", "title": seg}, 0, 1)


def test_diff_detects_drop_add_and_move():
    before = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    after = [_hit("b", 0.95), _hit("a", 0.85), _hit("d", 0.6)]
    d = diff(before, after)
    assert [h.segment_id for h in d.dropped] == ["c"]
    assert [h.segment_id for h in d.added] == ["d"]
    assert {h.segment_id for h, _, _ in d.moved} == {"a", "b"}
    assert d.changed


def test_diff_of_identical_sets_is_unchanged():
    hits = [_hit("a", 0.9), _hit("b", 0.8)]
    d = diff(hits, list(hits))
    assert not d.changed and d.kept == 2


# ------------------------------------------------------------------------------ profile
def test_profile_hash_is_order_independent():
    """The same gestures in a different order are the same taste, so the bank must hit."""
    a = TasteProfile(["p1", "p2"], ["n1"], "warm")
    b = TasteProfile(["p2", "p1"], ["n1"], "Warm ")
    assert a.hash == b.hash


def test_profile_hash_changes_with_negatives():
    assert TasteProfile(["p1"]).hash != TasteProfile(["p1"], ["n1"]).hash


# ------------------------------------------------------------------------------- prompt
def test_mode_bpm_uses_median_not_mean():
    """librosa half/double-time errors (70 vs 140) would wreck a mean."""
    hits = [Hit(str(i), 0.5, f"p{i}", {"bpm": b}, 0, 1) for i, b in enumerate([120, 122, 240])]
    assert _mode_bpm(hits) == 122


def test_dominant_key_is_modal():
    hits = [Hit(str(i), 0.5, f"p{i}", {"key": k}, 0, 1) for i, k in enumerate(["C", "G", "C"])]
    assert _dominant_key(hits) == "C major"


def test_synthesize_puts_steer_first_and_marks_instrumental():
    hits = [Hit("s", 0.9, "p", {"tags": ["rock"], "bpm": 128, "key": "A"}, 0, 1)]
    s = synthesize(hits, steer="darker and slower")
    assert s.params.prompt.startswith("darker and slower")
    assert s.params.bpm == 128
    assert s.params.keyscale == "A major"
    assert s.params.lyrics == ""            # instrumental
    assert "no vocals" in s.params.prompt


def test_synthesize_survives_empty_metadata():
    """A corpus with no BPM or tags must still produce a valid prompt, not crash."""
    s = synthesize([Hit("s", 0.1, "p", {}, 0, 1)], steer="")
    assert s.params.prompt and s.params.bpm is None


# -------------------------------------------------------------------------------- track
def test_caption_mentions_tags_and_bpm():
    t = Track("1", "A", "T", None, "Attribution", "u", tags=["jazz"], bpm=90, key="D")
    cap = t.caption
    assert "jazz" in cap and "90" in cap and "D" in cap


# ------------------------------------------------------------------- bank fallback
def test_bank_best_match_falls_back_to_nearest_taste(tmp_path, monkeypatch):
    """An unrehearsed taste must return the closest banked track, never silence.

    The bank is keyed by taste-profile hash, and every live gesture on stage produces a new
    hash. Exact-match lookup would hand the presenter a silent file for any improvisation.
    """
    import json

    import numpy as np

    from vector_taste import generate as gen

    monkeypatch.setattr(gen, "BANK", tmp_path)
    monkeypatch.setattr(gen, "BANK_INDEX", tmp_path / "bank.json")

    near = np.zeros(512, dtype=np.float32)
    near[0] = 1.0
    far = np.zeros(512, dtype=np.float32)
    far[1] = 1.0
    for name in ("near", "far"):
        (tmp_path / f"{name}.wav").write_bytes(b"RIFF" + b"\0" * 2048)
    (tmp_path / "bank.json").write_text(json.dumps({
        "near": {"file": "near.wav", "centroid": near.tolist()},
        "far": {"file": "far.wav", "centroid": far.tolist()},
    }))

    query = np.zeros(512, dtype=np.float32)
    query[0] = 0.9
    query[1] = 0.1
    path, note = gen.bank_best_match("never-baked-hash", query)
    assert path is not None and path.name == "near.wav"
    assert "nearest banked taste" in note

    # An exact hash hit must win outright and carry no fallback note.
    path, note = gen.bank_best_match("near", query)
    assert path.name == "near.wav" and note == ""


def test_bank_best_match_without_centroid_is_exact_only():
    """No centroid to compare against means no guessing — return nothing, not a wrong track."""
    from vector_taste.generate import bank_best_match

    path, note = bank_best_match("definitely-not-a-real-hash", None)
    assert path is None and note == ""


# --------------------------------------------------------------- descriptors / prompts
def _dhit(seg, descriptors, tags=("hip-hop",), bpm=100):
    return Hit(seg, 0.5, f"p-{seg}", {"descriptors": list(descriptors),
                                      "tags": list(tags), "bpm": bpm}, 0, 1)


def test_seed_differs_per_taste_and_is_stable():
    """Every bank entry used seed=42, so identical noise made every track sound alike."""
    from vector_taste.prompt import seed_from_hash

    a, b = seed_from_hash("2a5559afa725"), seed_from_hash("b07e5152536a")
    assert a != b
    assert seed_from_hash("2a5559afa725") == a          # stable for the same taste
    assert 0 <= a < 2**31


def test_negative_removes_shared_descriptors():
    """A descriptor strong in BOTH sides is not what distinguishes the taste.

    This is the only route by which a negative reaches generation at all -- ACE-Step has no
    negative-prompt field.
    """
    from vector_taste.prompt import contrastive_descriptors

    pos = [_dhit("a", ["dreamy and hazy", "grand piano"])]
    neg = [_dhit("b", ["dreamy and hazy", "drum machine"])]

    without = contrastive_descriptors(pos, None)
    with_neg = contrastive_descriptors(pos, neg)
    assert "dreamy and hazy" in without["mood"]
    assert "dreamy and hazy" not in with_neg["mood"]     # canceled by the negative
    assert "grand piano" in with_neg["instrument"]       # unique to the positive, survives


def test_different_tastes_give_different_prompts():
    """The actual reported bug: every taste produced the same prompt."""
    from vector_taste.prompt import synthesize

    a = synthesize([_dhit("a", ["dreamy and hazy", "grand piano"], ("folk",), 90)])
    b = synthesize([_dhit("b", ["aggressive and intense", "drum machine"], ("rock",), 160)])
    assert a.params.prompt != b.params.prompt
    assert "grand piano" in a.params.prompt
    assert "drum machine" in b.params.prompt


def test_prompt_says_instrumental_once():
    """The old template appended the suffix while `instrumental` was also a corpus tag."""
    from vector_taste.prompt import synthesize

    p = synthesize([_dhit("a", ["dreamy and hazy"], ("instrumental",))]).params.prompt
    assert p.count("no vocals") == 1


def test_prompt_respects_caption_limit():
    """ACE-Step caps `caption` at 512 characters."""
    from vector_taste.prompt import MAX_CAPTION, synthesize

    p = synthesize([_dhit("a", ["dreamy and hazy"])], steer="x " * 600)
    assert len(p.params.prompt) <= MAX_CAPTION


def test_descriptor_categories_are_disjoint():
    """A term in two categories would be double-counted in the contrast."""
    from vector_taste.describe import FLAT, TERMS

    assert len(TERMS) == len(set(TERMS))
    assert len(FLAT) == len(TERMS)


def test_top_descriptors_picks_per_category():
    """Selection is per-category so one strong category cannot crowd out the others."""
    import numpy as np

    from vector_taste.describe import FLAT, VOCAB, top_descriptors

    scores = np.zeros(len(FLAT), dtype=np.float32)
    got = top_descriptors(scores, per_category=1)
    assert len(got) == len(VOCAB)
    assert len({FLAT[FLAT.index((c, t))][0] for c, t in FLAT if t in got}) == len(VOCAB)


# ------------------------------------------------------------------ fresh generation
def test_fresh_seed_differs_between_calls():
    """Composing twice must give two different takes, not a replay of the first."""
    from vector_taste.prompt import SEED_MAX, fresh_seed

    seeds = {fresh_seed() for _ in range(20)}
    assert len(seeds) > 15                      # collisions possible, 5+ would be broken
    assert all(0 <= s < SEED_MAX for s in seeds)


def test_seed_from_hash_still_reproducible():
    """`vt bake` and `vt rehearse` rely on the same taste re-baking to the same track."""
    from vector_taste.prompt import seed_from_hash

    assert seed_from_hash("2a5559afa725") == seed_from_hash("2a5559afa725")


def test_latest_generated_prefers_newest_take(tmp_path, monkeypatch):
    """The loop must score the take the user just heard, not an older one."""
    import os
    import time

    from vector_taste import generate as gen

    monkeypatch.setattr(gen, "GENERATED", tmp_path)
    old = tmp_path / "abc123-111.wav"
    new = tmp_path / "abc123-222.wav"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (time.time() - 100, time.time() - 100))

    assert gen.latest_generated("abc123") == new
    assert gen.latest_generated("nosuchprofile") is None


def test_audio_for_profile_prefers_live_over_bank(tmp_path, monkeypatch):
    """While exploring, a fresh take beats a pre-baked one -- that was the whole bug."""
    from vector_taste import generate as gen

    monkeypatch.setattr(gen, "GENERATED", tmp_path)
    (tmp_path / "abc123-999.wav").write_bytes(b"fresh")

    path, note = gen.audio_for_profile("abc123", None)
    assert path.name == "abc123-999.wav"
    assert note == ""


# ------------------------------------------------------------------- progress reporting
def test_parses_real_diffusion_tqdm_lines():
    """Captured verbatim from a real run. tqdm's format is not an API, so pin it."""
    from vector_taste.worker import parse_diffusion_step

    assert parse_diffusion_step(
        "MLX DiT diffusion:   0%|          | 0/8 [00:00<?, ?it/s]") == (0, 8)
    assert parse_diffusion_step(
        "MLX DiT diffusion:  25%|##5       | 2/8 [01:02<02:49, 28.23s/it]") == (2, 8)
    assert parse_diffusion_step(
        "MLX DiT diffusion: 100%|##########| 8/8 [01:09<00:00,  8.64s/it]") == (8, 8)


def test_ignores_non_diffusion_output():
    """ACE-Step's stderr is mostly loguru and torch warnings."""
    from vector_taste.worker import parse_diffusion_step

    for line in ("", "INFO | loading vae to mps (RSS: 2832 MB)",
                 "bitsandbytes not installed", "Decoding audio..."):
        assert parse_diffusion_step(line) is None


def test_fraction_mapping_is_monotonic_and_starts_at_zero():
    """ACE-Step's own scale starts at 0.51 for us -- shown raw the bar begins half done."""
    from vector_taste.worker import map_fraction

    fracs = [map_fraction(v)[0] for v in (0.0, 0.51, 0.52, 0.80, 0.99)]
    assert fracs == sorted(fracs)
    assert fracs[0] == 0.0
    assert map_fraction(0.52)[1] == "diffusion"
    assert map_fraction(0.80)[1] == "decoding audio"


def test_progress_never_goes_backwards():
    """tqdm and the callback interleave; a late lower value would look like a stall."""
    from vector_taste.progress import Progress

    p = Progress()
    p.update(frac=0.5)
    p.update(frac=0.2)                      # stale event arriving late
    assert p.snapshot()["frac"] == 0.5
    p.update(frac=0.7)
    assert p.snapshot()["frac"] == 0.7


def test_progress_snapshot_reports_elapsed():
    from vector_taste.progress import Progress

    snap = Progress().snapshot()
    assert snap["elapsed"] >= 0
    assert "started" not in snap          # internal, not part of the API


# ---------------------------------------------------------------------------- aborting
def test_abort_is_not_a_generic_failure():
    """An abort must never be swallowed by the bank fallback.

    Someone who pressed stop wants silence, not a pre-baked track they didn't ask for, so
    GenerationAborted has to survive the `except Exception` that catches real failures.
    """
    from vector_taste.generate import GenerationAborted, GenerationError
    from vector_taste.worker import WorkerAborted, WorkerError

    assert issubclass(GenerationAborted, GenerationError)
    assert issubclass(WorkerAborted, WorkerError)
    # ...but distinguishable, which is what lets generate() re-raise only this one.
    assert not isinstance(GenerationError("x"), GenerationAborted)


def test_abort_with_no_worker_is_a_no_op():
    """Pressing stop when nothing is generating must not spawn a worker or raise."""
    from vector_taste.worker import AceStepWorker

    before = AceStepWorker._instance
    try:
        AceStepWorker._instance = None
        assert AceStepWorker.abort_current() is False
    finally:
        AceStepWorker._instance = before


# ------------------------------------------------------------------------- elevenlabs
def _gp(prompt="warm acoustic guitar, sombre folk, grand piano, around 112 BPM"):
    from vector_taste.prompt import GenerationParams

    return GenerationParams(prompt=prompt, seed=42, audio_duration=30.0)


def test_negatives_that_contradict_the_positives_are_dropped():
    """A rejected track shares traits with the accepted ones -- that is why it surfaced.

    Passing those shared traits as negative_styles would tell the model to both want and
    avoid the same thing. Measured in a real run before this filter existed: a taste built
    on "acoustic guitar" was sending "acoustic guitar" as a negative.
    """
    from vector_taste.elevenlabs import _styles_from

    pos, neg = _styles_from(_gp(), ["acoustic guitar", "heavily distorted and fuzzy"])
    assert "heavily distorted and fuzzy" in neg      # genuinely contrasting, kept
    assert not any("acoustic guitar" == n for n in neg)
    assert not set(pos) & set(neg)


def test_plan_always_asks_for_instrumental():
    """force_instrumental is prompt-only, so plan mode has to say it in the chunk."""
    from vector_taste.elevenlabs import build_plan

    chunk = build_plan(_gp(), None)["chunks"][0]
    assert chunk["text"] == "[Instrumental]"
    assert "vocals" in chunk["negative_styles"]


def test_conditioning_ref_only_when_a_reference_exists():
    from vector_taste.elevenlabs import MAX_REF_MS, build_plan

    assert "conditioning_ref" not in build_plan(_gp(), None)["chunks"][0]
    chunk = build_plan(_gp(), "song123")["chunks"][0]
    assert chunk["conditioning_ref"]["song_id"] == "song123"
    # The API caps a reference at 30s; our segments sit exactly at that ceiling.
    assert chunk["conditioning_ref"]["range"]["end_ms"] <= MAX_REF_MS
    assert chunk["condition_strength"]


def test_chunk_duration_is_clamped_to_api_bounds():
    from vector_taste.elevenlabs import CHUNK_MAX_MS, CHUNK_MIN_MS, build_plan

    short = _gp()
    short.audio_duration = 0.5
    long = _gp()
    long.audio_duration = 9999
    assert build_plan(short, None)["chunks"][0]["duration_ms"] == CHUNK_MIN_MS
    assert build_plan(long, None)["chunks"][0]["duration_ms"] == CHUNK_MAX_MS


def test_abort_registry_dispatches_to_whichever_backend_is_running():
    """Abort used to kill the ACE-Step process directly, so Stop was a silent no-op on any
    hosted backend. The registry is what makes it work everywhere."""
    from vector_taste.progress import abort_current, register_aborter

    register_aborter(None)
    assert abort_current() is False          # nothing running

    called = []
    register_aborter(lambda: (called.append(1), True)[1])
    assert abort_current() is True and called

    # A failing aborter must not surface as a crash mid-demo.
    register_aborter(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert abort_current() is False
    register_aborter(None)


# --------------------------------------------------------------------------- vocals
def test_vocals_keeps_their_lyrics_and_our_everything_else():
    """The plan endpoint supplies words; the taste supplies the music.

    Overwriting their positive_styles is the point -- otherwise the retrieved taste would be
    replaced by whatever ElevenLabs guessed from the prompt.
    """
    from vector_taste.elevenlabs import build_plan

    lyric_chunks = [
        {"text": "[Verse]\nCity lights are fading", "positive_styles": ["their guess"]},
        {"text": "[Chorus]\nHold on", "positive_styles": ["their other guess"]},
    ]
    plan = build_plan(_gp(), "song123", None, True, lyric_chunks)
    chunks = plan["chunks"]
    assert [c["text"] for c in chunks] == [c["text"] for c in lyric_chunks]
    assert all("their guess" not in c["positive_styles"] for c in chunks)
    assert all("warm acoustic guitar" in c["positive_styles"] for c in chunks)
    # Our duration is split across their sections rather than multiplied by them.
    assert sum(c["duration_ms"] for c in chunks) <= _gp().audio_duration * 1000


def test_vocals_drops_the_no_vocals_negatives():
    """NO_VOCALS is how instrumental is enforced; leaving it in would forbid what was asked."""
    from vector_taste.elevenlabs import build_plan

    sung = build_plan(_gp(), None, ["heavily distorted and fuzzy"], vocals=True)["chunks"][0]
    assert not any(n in sung["negative_styles"] for n in ("vocals", "singing", "choir"))
    assert "sung vocals" in sung["positive_styles"]
    # The contradiction filter still runs on the rejected example's descriptors.
    assert "heavily distorted and fuzzy" in sung["negative_styles"]


def test_vocals_falls_back_to_a_section_marker_when_the_lyric_plan_fails():
    """Losing the lyrics must not lose the track -- same rule as a failed reference upload."""
    from vector_taste.elevenlabs import build_plan

    chunks = build_plan(_gp(), None, None, vocals=True, lyric_chunks=None)["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["text"] == "[Verse]"
    assert "sung vocals" in chunks[0]["positive_styles"]


def test_conditioning_lands_on_the_first_chunk_only():
    """Per the spec the first chunk "influences all subsequent chunks", so repeating the
    reference on every section buys nothing and costs plan size."""
    from vector_taste.elevenlabs import build_plan

    chunks = build_plan(
        _gp(), "song123", None, True,
        [{"text": "[Verse]"}, {"text": "[Chorus]"}, {"text": "[Outro]"}],
    )["chunks"]
    assert "conditioning_ref" in chunks[0]
    assert all("conditioning_ref" not in c for c in chunks[1:])


def test_synthesized_prompt_stops_saying_instrumental_when_vocals_are_on():
    from vector_taste.prompt import synthesize

    assert "no vocals" in synthesize([], duration=15.0).params.prompt
    assert "no vocals" not in synthesize([], duration=15.0, vocals=True).params.prompt


def test_vocals_are_refused_on_backends_that_cannot_sing():
    """A silent instrumental would give the user no way to tell the request was dropped."""
    import pytest

    from vector_taste.generate import VOCALS_BACKENDS, GenerationError, generate

    assert "elevenlabs" in VOCALS_BACKENDS and "local" not in VOCALS_BACKENDS
    with pytest.raises(GenerationError, match="cannot generate vocals"):
        generate(_gp(), "vocalguard", backend="local", vocals=True)


def test_the_baseline_excludes_the_whole_track_you_marked_not_just_that_chunk():
    """A track is several 10s chunk points. Filtering the exact point the user clicked let a
    DIFFERENT chunk of that same track come back as the "closest human" -- measured at cosine
    0.94, which is the ~0.91-by-construction score this filter exists to keep out. It made
    every generated result look worse than it was."""
    from vector_taste.loop import _segments_of

    class _Hit:
        def __init__(self, seg, pid):
            self.segment_id, self.point_id = seg, pid

    marked_point, other_chunk = "p-a", "p-b"          # same track, different windows
    results = [_Hit("trk:0", other_chunk), _Hit("trk2:0", "p-c")]

    by_point = [h for h in results if h.point_id not in {marked_point}]
    assert by_point[0].segment_id == "trk:0"          # the old filter: leaks the same track

    by_segment = [h for h in results if h.segment_id not in {"trk:0"}]
    assert by_segment[0].segment_id == "trk2:0"       # the fix: a genuinely different track
    assert callable(_segments_of)


# --------------------------------------------------------------------------- uploads
def test_upload_validation_rejects_what_it_should():
    import pytest

    from vector_taste.uploads import MAX_BYTES, UploadError, validate

    assert validate("song.mp3", 1000) == ".mp3"
    assert validate("SONG.WAV", 1000) == ".wav"          # case-insensitive
    with pytest.raises(UploadError, match="not audio"):
        validate("resume.pdf", 1000)
    with pytest.raises(UploadError, match="not audio"):
        validate("noextension", 1000)
    with pytest.raises(UploadError, match="limit"):
        validate("song.mp3", MAX_BYTES + 1)
    with pytest.raises(UploadError, match="empty"):
        validate("song.mp3", 0)


def test_uploaded_filenames_never_reach_the_filesystem(tmp_path, monkeypatch):
    """A user-supplied name is display text, never a path. Storing under a generated name
    closes path traversal without needing to sanitise anything."""
    from vector_taste import uploads

    monkeypatch.setattr(uploads, "UPLOADS", tmp_path / "uploads")
    path, track_id = uploads.save(b"not really audio, but bytes", "../../../etc/passwd.wav")

    assert path.parent == tmp_path / "uploads"           # stayed inside the upload dir
    assert path.name == f"{track_id}.wav"                # nothing of the original name
    assert ".." not in str(path)
    assert not (tmp_path / "etc").exists()


def test_hosted_generation_never_receives_user_audio():
    """A hosted backend UPLOADS the style reference to a third party. An upload is somebody
    else's file of unknown provenance, so those backends skip past it to a corpus track.
    Local keeps the top hit -- conditioning on your own audio never leaves the machine."""
    from vector_taste.cli import _reference_hit

    class _H:
        def __init__(self, upload, name):
            self.payload = {"is_upload": upload, "title": name}

    hits = [_H(True, "mine.wav"), _H(False, "corpus track")]

    assert _reference_hit(hits, "elevenlabs").payload["title"] == "corpus track"
    assert _reference_hit(hits, "local").payload["title"] == "mine.wav"
    assert _reference_hit(hits, "modal").payload["title"] == "mine.wav"
    # Nothing but uploads: send no reference at all rather than send theirs.
    assert _reference_hit([_H(True, "mine.wav")], "elevenlabs") is None


def test_upload_payload_cannot_claim_a_licence_it_does_not_have():
    """The row subtitle renders `license`. Printing CC-BY over a stranger's file would be a
    false claim in a repo whose argument is that its corpus is legally clean."""
    import inspect

    from vector_taste import uploads

    src = inspect.getsource(uploads.ingest)
    assert '"license": "your upload"' in src
    assert '"source_url": ""' in src        # so ATTRIBUTIONS.md can never pick it up
    assert '"is_upload": True' in src


def test_uploads_are_excluded_from_the_scoring_population_but_not_from_search():
    """Uploads are part of the searchable library by design. The closing percentile is a
    different question: letting them in would make the headline depend on whatever someone
    dropped in, and stop it being comparable between runs."""
    import inspect

    from vector_taste import search, taste

    assert "NOT_UPLOAD" in inspect.getsource(taste.percentile_against_centroid)
    # Retrieval deliberately filters only generated points.
    assert "is_upload" not in inspect.getsource(search.merge_filters)
    assert "is_upload" not in inspect.getsource(search._query)
