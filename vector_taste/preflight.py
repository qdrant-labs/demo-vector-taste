"""Pre-demo checklist. One command, run before walking on stage.

Each check prints a fix rather than a stack trace. The point is to fail in the green room,
not on the projector.
"""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

from .config import AUDIO, BANK, CLAP_MODEL, COLLECTION, DATA, get_client, is_cloud

OK, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


def _row(status: str, name: str, detail: str = "", fix: str = "") -> tuple[str, str, str, str]:
    return (status, name, detail, fix)


def check_qdrant():
    try:
        client = get_client()
        info = client.get_collection(COLLECTION)
        n = info.points_count
        if n == 0:
            return _row(FAIL, "qdrant", "collection is empty", "run: uv run vt bootstrap")
        where = "Qdrant Cloud" if is_cloud() else "local container"
        return _row(OK, "qdrant", f"{n} points on {where}")
    except Exception as exc:  # noqa: BLE001
        return _row(FAIL, "qdrant", str(exc)[:70], "run: ./scripts/qdrant_up.sh")


def check_no_generated():
    """Generated points must be purged or the finale percentile drifts between runs."""
    from .store import count

    try:
        n = count(only_generated=True)
        if n:
            return _row(FAIL, "clean state", f"{n} generated points present",
                        "run: uv run vt reset")
        return _row(OK, "clean state", "no generated points")
    except Exception as exc:  # noqa: BLE001
        return _row(WARN, "clean state", str(exc)[:60])


def check_models():
    """Model must be in the local HF cache, or first use on stage hits the network."""
    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = cache / "hub" if (cache / "hub").exists() else cache
    slug = "models--" + CLAP_MODEL.replace("/", "--")
    if (hub / slug).exists():
        return _row(OK, "clap model", f"cached: {CLAP_MODEL}")
    return _row(FAIL, "clap model", "not in local cache",
                f"run once online: uv run vt search --text test")


def check_audio_files():
    if not AUDIO.exists():
        return _row(FAIL, "corpus audio", "audio/ missing", "run: uv run vt fetch")
    n = sum(1 for _ in AUDIO.rglob("*.mp3"))
    if n == 0:
        return _row(FAIL, "corpus audio", "no mp3 files", "run: uv run vt fetch")
    return _row(OK, "corpus audio", f"{n} files")


def check_bank():
    from .generate import bank_status

    st = bank_status()
    if st["entries"] == 0:
        return _row(WARN, "generation bank", "empty", "run: uv run vt bake")
    if not st["complete"]:
        missing = st["entries"] - st["files_present"]
        return _row(FAIL, "generation bank", f"{missing} entries missing audio",
                    "run: uv run vt bake")
    return _row(OK, "generation bank", f"{st['entries']} entries")


def check_profiles():
    profiles = sorted(DATA.glob("taste_*.json"))
    if not profiles:
        return _row(WARN, "taste profiles", "none saved", "run: uv run vt taste --pos ...")
    return _row(OK, "taste profiles", f"{len(profiles)} saved")


def check_offline():
    """Confirm nothing on the demo path needs the internet.

    Checks that the pieces are local rather than that the network is down — you may well be
    online while rehearsing.
    """
    problems = []
    if is_cloud():
        problems.append("QDRANT_URL points at Cloud")
    if os.getenv("GEN_BACKEND") in ("replicate", "modal"):
        problems.append(f"GEN_BACKEND={os.getenv('GEN_BACKEND')} needs network")
    if problems:
        return _row(WARN, "offline safety", "; ".join(problems),
                    "for stage: unset QDRANT_URL/API_KEY, GEN_BACKEND=bank")
    return _row(OK, "offline safety", "all demo-path components are local")


def check_audio_output():
    """Conference AV commonly captures the display but not system audio.

    We cannot verify audibility programmatically, so this prompts the human to check the
    one thing that silently ruins a music demo.
    """
    if shutil.which("SwitchAudioSource"):
        import subprocess

        try:
            dev = subprocess.run(
                ["SwitchAudioSource", "-c"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            return _row(WARN, "audio output", f"device: {dev}",
                        "PLAY A TRACK OUT LOUD before you start")
        except Exception:  # noqa: BLE001
            pass
    return _row(WARN, "audio output", "cannot verify automatically",
                "PLAY A TRACK OUT LOUD through the venue PA before you start")


def check_port(port: int = 8000):
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return _row(WARN, "ui port", f"{port} already in use", f"kill it or use --port")
    return _row(OK, "ui port", f"{port} free")


CHECKS = [
    check_qdrant,
    check_no_generated,
    check_models,
    check_audio_files,
    check_bank,
    check_profiles,
    check_offline,
    check_port,
    check_audio_output,
]


def run_preflight(check_audio: bool = True) -> bool:
    checks = CHECKS if check_audio else [c for c in CHECKS if c is not check_audio_output]
    rows = []
    for c in checks:
        try:
            rows.append(c())
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(FAIL, c.__name__, str(exc)[:60]))

    print()
    print("  PREFLIGHT")
    print("  " + "-" * 76)
    for status, name, detail, fix in rows:
        mark = {OK: "  ok ", WARN: " warn", FAIL: " FAIL", SKIP: " skip"}[status]
        print(f"  [{mark}] {name:<16} {detail[:46]}")
        if fix and status in (FAIL, WARN):
            print(f"           -> {fix}")
    print("  " + "-" * 76)

    failed = [r for r in rows if r[0] == FAIL]
    if failed:
        print(f"  {len(failed)} BLOCKING issue(s). Fix before going on stage.\n")
        return False
    warns = [r for r in rows if r[0] == WARN]
    print(f"  ready{f' ({len(warns)} warnings — read them)' if warns else ''}\n")
    return True
