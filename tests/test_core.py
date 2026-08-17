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
    licence, so it is rejected too.
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
