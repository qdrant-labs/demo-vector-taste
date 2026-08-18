"""Corpus download.

Audio is never committed to the repo — it is fetched here and license-filtered at ingest.
Downloads are resumable and skip work already done, so re-running is cheap.
"""

from __future__ import annotations

import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .config import AUDIO, RAW
from .corpus import FMA_ARCHIVES, FMA_METADATA_URL, FMA_SMALL_URL
from .timing import stage

FILES = {
    # Exact content-length, not an estimate. A guess that is LARGER than the real file makes
    # a completed download look unfinished, and the resume then asks for a range starting at
    # EOF -- which the server answers 416. Handled below too, belt and braces.
    "fma_metadata.zip": (FMA_METADATA_URL, 358_412_441),
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
        # 416 on a resume means we asked for bytes past the end -- i.e. the file is already
        # complete and our `expected` was simply wrong. That is a success, not a failure.
        if have and r.status_code == 416:
            print(f"  {dest.name}: already complete ({have / 1e9:.2f} GB)")
            return dest
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


class HttpRangeFile(io.RawIOBase):
    """A seekable file over HTTP Range requests.

    `zipfile.ZipFile` only needs read/seek/tell/seekable, so this is enough to open a remote
    archive and pull individual members out of it. That matters here: fma_large.zip is 100GB
    and we want 8,780 of its 106,574 files, so downloading the whole thing would discard 92%
    of the bytes -- and would not fit on the disk in the first place.

    Measured against the real archive: the central directory costs 4 requests and 8.7MB, and
    one track costs ~1MB.
    """

    def __init__(self, url: str, client: httpx.Client, size: int | None = None):
        self.url, self._c, self._pos = url, client, 0
        if size is not None:          # already probed; skip a redundant round trip
            self.size = size
            return
        r = client.head(url, follow_redirects=True, timeout=60)
        r.raise_for_status()
        if r.headers.get("accept-ranges") != "bytes":
            raise RuntimeError(f"{url} does not advertise byte ranges; use --full-archive")
        self.size = int(r.headers["content-length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, min(self.size, base + offset))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        end = self._pos + n - 1
        last = None
        for attempt in range(4):          # a 100GB read over thousands of requests will blip
            try:
                r = self._c.get(
                    self.url,
                    headers={"Range": f"bytes={self._pos}-{end}"},
                    follow_redirects=True,
                    timeout=120,
                )
                r.raise_for_status()
                self._pos += len(r.content)
                return r.content
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                last = exc
                import time

                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"range read failed after 4 attempts: {last}")


def _read_member(fh: HttpRangeFile, info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress ONE zip member in a single ranged read.

    The central directory already told us where this member starts and how many compressed
    bytes it has; all the local header adds is the length of its name and extra fields. So
    read the header plus a generous slack plus the payload in one request, then slice.

    Verified against the CRC in the central directory, which is what makes a partial or
    mis-offset read fail loudly instead of writing a corrupt mp3.
    """
    import binascii
    import bz2
    import lzma
    import struct
    import zlib

    SLACK = 512
    fh.seek(info.header_offset)
    blob = fh.read(30 + SLACK + info.compress_size)
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"no local header for {info.filename} at {info.header_offset}")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    if start + info.compress_size > len(blob):     # unusually large extra field
        fh.seek(info.header_offset + start)
        raw = fh.read(info.compress_size)
    else:
        raw = blob[start : start + info.compress_size]

    # Measured: every one of fma_large's 106,574 members is BZIP2, which is why
    # extract_audio() cannot shell out to macOS's unzip either. The other branches are
    # cheap insurance if FMA ever re-packs.
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
    elif info.compress_type == zipfile.ZIP_BZIP2:
        data = bz2.decompress(raw)
    elif info.compress_type == zipfile.ZIP_LZMA:
        # zip wraps LZMA in a 4-byte header carrying the properties length.
        props_len = struct.unpack("<H", raw[2:4])[0]
        props = raw[4 : 4 + props_len]
        dec = lzma.LZMADecompressor(
            lzma.FORMAT_RAW,
            filters=[lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)],
        )
        data = dec.decompress(raw[4 + props_len :])
    else:
        raise RuntimeError(f"unsupported compression {info.compress_type} for {info.filename}")

    if binascii.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise RuntimeError(f"CRC mismatch for {info.filename}")
    return data


def fetch_selective(
    subset: str, dest: Path = AUDIO, workers: int = 8, limit: int | None = None
) -> int:
    """Pull ONLY the permissively-licensed tracks out of a remote FMA archive.

    Idempotent and resumable: anything already on disk is skipped, so the tracks fetched by
    an earlier run (or by the whole-archive path) are never fetched twice.

    `workers` is deliberately modest. os.unil.cloud.switch.ch is a Swiss academic host
    doing us a favour, and this issues one request per track.
    """
    from .corpus import fma_audio_path, load_fma_metadata

    url = FMA_ARCHIVES[subset][0]
    rows = load_fma_metadata(subset=subset)
    wanted = {r["track_id"]: fma_audio_path(r["track_id"]) for r in rows}
    todo = {tid: out for tid, out in wanted.items() if not out.exists()}
    if limit:
        todo = dict(list(todo.items())[:limit])
    print(f"  {len(wanted)} usable tracks in '{subset}'; {len(todo)} to fetch")
    if not todo:
        return 0

    with httpx.Client(timeout=120) as client:
        root = HttpRangeFile(url, client)
        archive_size = root.size
        zf = zipfile.ZipFile(root)
        # Map zip members back to track ids: fma_large/<nnn>/<id>.mp3
        by_id = {}
        for info in zf.infolist():
            if info.filename.endswith(".mp3"):
                by_id[Path(info.filename).stem.lstrip("0") or "0"] = info
        print(f"  archive lists {len(by_id)} tracks")

        missing = [t for t in todo if t not in by_id]
        if missing:
            print(f"  note: {len(missing)} wanted tracks are absent from the archive")

        done = [0]
        failed: list[str] = []
        # One client and one cursor PER THREAD, reused across that thread's tracks. Building
        # them per track cost a redundant HEAD each (7,630 of them) and put that unretried
        # request on the critical path -- which is exactly where a DNS blip killed the first
        # full run. It also must NOT re-open the ZipFile: re-reading the 8.7MB central
        # directory per track would be 76GB across the corpus.
        tl = threading.local()

        def handle() -> HttpRangeFile:
            if getattr(tl, "fh", None) is None:
                tl.client = httpx.Client(timeout=120)
                tl.fh = HttpRangeFile(url, tl.client, size=archive_size)
            return tl.fh

        def grab(tid: str) -> int:
            info = by_id.get(tid)
            if info is None:
                return 0
            out = todo[tid]
            out.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                try:
                    data = _read_member(handle(), info)
                    break
                except Exception:  # noqa: BLE001 - network/DNS blips over ~8k requests
                    # Drop the connection so the next attempt re-resolves and re-dials.
                    if getattr(tl, "client", None) is not None:
                        tl.client.close()
                    tl.fh = tl.client = None
                    if attempt == 3:
                        failed.append(tid)
                        return 0
                    time.sleep(2 * (attempt + 1))
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(out)                # atomic: a killed run never leaves a half file
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"\r  fetched {done[0]}/{len(todo)}", end="", flush=True)
            return len(data)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(grab, list(todo)))

    print(f"\r  fetched {done[0]} tracks ({sum(sizes) / 1e9:.2f} GB)")
    if failed:
        print(f"  {len(failed)} tracks failed after retries; re-run to pick them up")
    return done[0]


def fetch_all(
    audio: bool = True,
    subset: str = "small",
    selective: bool = True,
    workers: int = 8,
    limit: int | None = None,
) -> None:
    """Metadata, then audio.

    `selective` pulls only the CC0/CC-BY tracks straight out of the remote archive. For
    `large` that is the difference between ~9GB and 100GB, and the 100GB does not fit on a
    laptop. `--full-archive` restores the download-and-extract path.
    """
    with stage("fetch.metadata"):
        download(FMA_METADATA_URL, RAW / "fma_metadata.zip", FILES["fma_metadata.zip"][1])
        tracks = RAW / "fma_metadata" / "tracks.csv"
        if not tracks.exists():
            with zipfile.ZipFile(RAW / "fma_metadata.zip") as zf:
                zf.extract("fma_metadata/tracks.csv", RAW)
            print("  extracted tracks.csv")

    if not audio:
        return

    if selective:
        with stage("fetch.audio.selective") as m:
            m["subset"] = subset
            m["new_files"] = fetch_selective(subset, AUDIO, workers=workers, limit=limit)
        return

    url, size = FMA_ARCHIVES[subset]
    with stage("fetch.audio") as m:
        zip_path = RAW / f"fma_{subset}.zip"
        download(url, zip_path, size)
        m["new_files"] = extract_audio(zip_path, AUDIO)
