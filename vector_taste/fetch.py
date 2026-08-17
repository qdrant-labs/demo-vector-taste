"""Corpus download.

Audio is never committed to the repo — it is fetched here and license-filtered at ingest.
Downloads are resumable and skip work already done, so re-running is cheap.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx

from .config import AUDIO, RAW
from .corpus import FMA_METADATA_URL, FMA_SMALL_URL
from .timing import stage

FILES = {
    "fma_metadata.zip": (FMA_METADATA_URL, 342 * 1024 * 1024),
    "fma_small.zip": (FMA_SMALL_URL, 7_679_594_875),
}


def download(url: str, dest: Path, expected: int | None = None) -> Path:
    """Resumable download with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    if expected and have >= expected:
        print(f"  {dest.name}: already complete ({have / 1e9:.1f} GB)")
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    if have:
        print(f"  {dest.name}: resuming at {have / 1e9:.2f} GB")

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        # A server that ignores Range replies 200; restart rather than corrupt the file.
        if have and r.status_code == 200:
            mode, have = "wb", 0
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + have
        done = have
        with dest.open(mode) as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {dest.name}: {done / 1e9:5.2f}/{total / 1e9:5.2f} GB "
                          f"({pct:5.1f}%)", end="", flush=True)
        print()
    return dest


def extract_audio(zip_path: Path, dest: Path) -> int:
    """Extract FMA audio.

    Uses Python's zipfile rather than the `unzip` binary: FMA archives are bzip2-compressed
    and macOS's system unzip refuses them ("need PK compat. v4.6").
    """
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".mp3")]
        for i, m in enumerate(members, 1):
            # Flatten fma_small/<nnn>/<id>.mp3 -> audio/<nnn>/<id>.mp3
            parts = Path(m).parts
            if len(parts) < 2:
                continue
            out = dest / parts[-2] / parts[-1]
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, out.open("wb") as fh:
                    fh.write(src.read())
                n += 1
            if i % 500 == 0:
                print(f"\r  extracting: {i}/{len(members)}", end="", flush=True)
    print(f"\r  extracted {n} new files ({len(members)} total)")
    return n


def fetch_all(audio: bool = True) -> None:
    with stage("fetch.metadata"):
        download(FMA_METADATA_URL, RAW / "fma_metadata.zip", FILES["fma_metadata.zip"][1])
        tracks = RAW / "fma_metadata" / "tracks.csv"
        if not tracks.exists():
            with zipfile.ZipFile(RAW / "fma_metadata.zip") as zf:
                zf.extract("fma_metadata/tracks.csv", RAW)
            print("  extracted tracks.csv")

    if not audio:
        return

    with stage("fetch.audio") as m:
        zip_path = RAW / "fma_small.zip"
        download(FMA_SMALL_URL, zip_path, FILES["fma_small.zip"][1])
        m["new_files"] = extract_audio(zip_path, AUDIO)
