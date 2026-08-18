"""Corpus loading: fetch, license-filter, describe.

The license filter is the load-bearing part of this module. FMA stores licenses as free
text with dozens of spellings across jurisdictions ("Attribution-Noncommercial-Share Alike
3.0 United States", "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0", ...),
so classification is deny-list-first: anything mentioning NonCommercial, ShareAlike, or
NoDerivatives is rejected regardless of how it's spelled, and only then do we require an
affirmative CC0 / Public Domain / Attribution match.

Why all three clauses are excluded, not just NC:
  ND - using a clip as a generation style reference arguably creates a derivative work.
  SA - would propagate share-alike obligations onto generated output.
  NC - this repo is published by a company.

To add your own source, write a function returning list[Track] with `license` and
`source_url` populated, and register it in SOURCES. Nothing downstream is FMA-specific.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import AUDIO, RAW

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"

# The archives are NESTED: fma_small ⊂ fma_medium ⊂ fma_large, all 30s clips. Sizes measured
# with a HEAD request, and they are the reason `vt fetch --selective` exists -- we want 8,780
# of fma_large's 106,574 files, so downloading the whole 100GB would discard 92% of it.
FMA_ARCHIVES = {
    "small": ("https://os.unil.cloud.switch.ch/fma/fma_small.zip", 7_679_594_875),
    "medium": ("https://os.unil.cloud.switch.ch/fma/fma_medium.zip", 23_800_000_000),
    "large": ("https://os.unil.cloud.switch.ch/fma/fma_large.zip", 100_306_112_191),
}

# `set.subset` in tracks.csv is an ORDERED category: a track carries the SMALLEST subset it
# belongs to. FMA's own utilities select with `<=`, so "the large subset" means small +
# medium + large. Matching a single value exactly (as this once did) silently returned only
# the 6,435 large-only tracks instead of all 8,780 usable ones.
SUBSET_ORDER = ["small", "medium", "large"]

# Order matters: deny wins over allow.
_DENY = re.compile(r"non-?commercial|share.?alike|no.?deriv|sampling", re.IGNORECASE)
_ALLOW = re.compile(r"\b(cc0|public\s?domain|attribution)\b", re.IGNORECASE)


def is_permissive(license_str: str | None) -> bool:
    """True only for CC0 / Public Domain / plain CC-BY. Empty or unknown -> False.

    Defaulting unknown licenses to False is deliberate: a track with missing license
    metadata is not a track we can safely redistribute or feed to a generative model.
    """
    if not license_str:
        return False
    s = license_str.strip()
    if not s:
        return False
    return bool(_ALLOW.search(s)) and not _DENY.search(s)


def license_short(license_str: str) -> str:
    """Normalize a messy license string to a short tag for display and attribution."""
    s = license_str.lower()
    if "cc0" in s:
        return "CC0-1.0"
    if "public domain" in s:
        return "Public Domain"
    m = re.search(r"(\d\.\d)", s)
    return f"CC-BY-{m.group(1)}" if m else "CC-BY"


@dataclass
class Track:
    track_id: str
    artist: str
    title: str
    path: Path
    license: str
    source_url: str
    tags: list[str] = field(default_factory=list)
    bpm: int | None = None
    key: str | None = None

    @property
    def caption(self) -> str:
        """Short natural-language description, embedded into the `text` vector.

        CLAP's text tower was trained on captions like "hip hop music with a heavy beat",
        so a genre-and-mood phrase retrieves far better than a bare tag list.
        """
        bits = [t for t in self.tags if t]
        base = ", ".join(bits[:4]) if bits else "music"
        parts = [f"{base} music" if not base.endswith("music") else base]
        if self.bpm:
            parts.append(f"{self.bpm} BPM")
        if self.key:
            parts.append(f"in {self.key}")
        return " ".join(parts)


def _extract(zip_path: Path, members: list[str], dest: Path) -> None:
    """Extract specific members.

    Uses Python's zipfile rather than the `unzip` binary: FMA's archives are bzip2-
    compressed (compress_type 12) and macOS's system unzip refuses them with
    "need PK compat. v4.6".
    """
    with zipfile.ZipFile(zip_path) as zf:
        for m in members:
            zf.extract(m, dest)


def load_fma_metadata(limit: int | None = None, subset: str = "small") -> list[dict]:
    """Parse FMA tracks.csv, keeping only permissively licensed rows in `subset` and below.

    CUMULATIVE, per SUBSET_ORDER: `subset="large"` yields small + medium + large. That is
    what FMA means by a subset, and it is what makes the counts come out at 1,005 / 2,345 /
    8,780 rather than 1,005 / 1,340 / 6,435.

    Reads with csv rather than pandas: tracks.csv is 248MB with a two-row header, and we
    only need six columns.
    """
    if subset not in SUBSET_ORDER:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSET_ORDER}")
    wanted = set(SUBSET_ORDER[: SUBSET_ORDER.index(subset) + 1])
    meta_zip = RAW / "fma_metadata.zip"
    tracks_csv = RAW / "fma_metadata" / "tracks.csv"
    if not tracks_csv.exists():
        if not meta_zip.exists():
            raise FileNotFoundError(f"missing {meta_zip}; run `vt fetch` first")
        _extract(meta_zip, ["fma_metadata/tracks.csv"], RAW)

    with tracks_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        h1, h2 = next(reader), next(reader)
        next(reader)  # third row is a units/dtype row, not data

        def col(group: str, name: str) -> int:
            for i, (a, b) in enumerate(zip(h1, h2, strict=False)):
                if a == group and b == name:
                    return i
            raise KeyError(f"({group},{name}) not in tracks.csv")

        c_lic, c_sub = col("track", "license"), col("set", "subset")
        c_artist, c_title = col("artist", "name"), col("track", "title")
        c_gtop = col("track", "genre_top")

        out: list[dict] = []
        for row in reader:
            if not row or row[c_sub] not in wanted:
                continue
            lic = row[c_lic]
            if not is_permissive(lic):
                continue
            tid = row[0]
            genres = []
            if c_gtop and row[c_gtop]:
                genres.append(row[c_gtop])
            out.append(
                {
                    "track_id": tid,
                    "artist": (row[c_artist] or "Unknown artist").strip(),
                    "title": (row[c_title] or f"Track {tid}").strip(),
                    "license": lic,
                    "license_short": license_short(lic),
                    "tags": [g.strip().lower() for g in genres if g.strip()],
                    "source_url": f"https://freemusicarchive.org/music/{tid}",
                }
            )
            if limit and len(out) >= limit:
                break
    return out


def fma_audio_path(track_id: str) -> Path:
    """FMA lays audio out as audio/<first 3 digits of zero-padded id>/<id>.mp3."""
    tid = f"{int(track_id):06d}"
    return AUDIO / tid[:3] / f"{tid}.mp3"


def analyze(path: Path) -> tuple[int | None, str | None]:
    """Estimate BPM and key. No CC source ships either, so we compute them.

    Both are payload/filter metadata only. BPM is also passed to ACE-Step at generation.
    """
    import librosa
    import numpy as np

    from .config import SAMPLE_RATE

    try:
        y, sr = librosa.load(str(path), sr=22_050, mono=True, duration=30)
    except Exception:
        return None, None
    if not len(y):
        return None, None

    bpm = None
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        t = float(np.atleast_1d(tempo)[0])
        if 30 <= t <= 300:
            bpm = int(round(t))
    except Exception:
        pass

    key = None
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = names[int(chroma.argmax())]
    except Exception:
        pass

    del SAMPLE_RATE  # analysis runs at 22.05k; embedding is the only 48k consumer
    return bpm, key


SOURCES = {"fma": load_fma_metadata}
