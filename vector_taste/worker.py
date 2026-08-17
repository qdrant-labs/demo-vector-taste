"""Client for the resident ACE-Step worker (see scripts/acestep_worker.py).

Spawned lazily with ACE-Step's own interpreter and kept alive for the process lifetime, so
the ~60-100s model load and MLX graph compilation is paid once rather than per track.

Communication is line-oriented JSON over stdin/stdout, with a `@@VT@@` sentinel on the way
back because ACE-Step's dependencies write freely to both streams (loguru, tqdm, torch
warnings) and stdout cannot be assumed to be clean.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path

from .config import ROOT

log = logging.getLogger("vector_taste.worker")

SENTINEL = "@@VT@@"
WORKER = ROOT / "scripts" / "acestep_worker.py"

# Generous: a cold worker loads ~9GB of checkpoints and compiles MLX graphs.
READY_TIMEOUT = float(os.getenv("VT_WORKER_READY_TIMEOUT", "900"))
JOB_TIMEOUT = float(os.getenv("VT_WORKER_JOB_TIMEOUT", "900"))


class WorkerError(RuntimeError):
    pass


def acestep_dir() -> Path:
    """ACE-Step's repo root. Its `acestep` package and checkpoints/ both live here."""
    # Guard the env var: Path("") / ".venv" collapses to a RELATIVE ".venv/bin/python",
    # which matches THIS project's interpreter and would run in the wrong environment.
    env = os.getenv("ACESTEP_DIR")
    return Path(env).expanduser().resolve() if env else (ROOT / ".acestep").resolve()


def acestep_python() -> Path | None:
    py = acestep_dir() / ".venv" / "bin" / "python"
    return py if py.is_file() else None


class AceStepWorker:
    """A resident ACE-Step subprocess. Not thread-safe by design; guarded by a lock."""

    _instance: AceStepWorker | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        py = acestep_python()
        if py is None:
            raise WorkerError(
                f"ACE-Step is not installed at {acestep_dir()}. "
                "Run ./scripts/acestep_setup.sh, or use GEN_BACKEND=bank."
            )
        self._proc = subprocess.Popen(  # noqa: S603
            [str(py), str(WORKER)],
            cwd=str(acestep_dir()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # ACE-Step's logging; kept off the protocol stream
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        self._job_lock = threading.Lock()
        log.info("ACE-Step worker starting (this takes a minute or two on first run)")
        ready = self._read_event(READY_TIMEOUT)
        if not ready.get("ok"):
            raise WorkerError(ready.get("error") or "worker failed to start")
        log.info("ACE-Step worker ready in %ss", ready.get("seconds"))

    # -------------------------------------------------------------------- protocol
    def _read_event(self, timeout: float) -> dict:
        """Read until the next sentinel line. Non-protocol output is ignored."""
        result: dict = {}
        done = threading.Event()

        def reader():
            nonlocal result
            for line in self._proc.stdout:
                if line.startswith(SENTINEL):
                    try:
                        result = json.loads(line[len(SENTINEL):].strip())
                    except json.JSONDecodeError:
                        continue
                    done.set()
                    return
            done.set()  # stream closed

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        if not done.wait(timeout):
            raise WorkerError(f"ACE-Step worker timed out after {timeout:.0f}s")
        if not result:
            raise WorkerError("ACE-Step worker exited unexpectedly (check its stderr)")
        return result

    def generate(self, job: dict) -> Path:
        """Run one job. Blocks until the worker returns audio."""
        with self._job_lock:
            if self._proc.poll() is not None:
                AceStepWorker._instance = None
                raise WorkerError("ACE-Step worker has exited; it will respawn next call")
            self._proc.stdin.write(json.dumps(job) + "\n")
            self._proc.stdin.flush()
            ev = self._read_event(JOB_TIMEOUT)

        if not ev.get("ok"):
            raise WorkerError(ev.get("error") or "generation failed")
        path = Path(ev["path"])
        if not path.exists():
            raise WorkerError(f"worker reported {path}, which does not exist")
        log.info("generated %s in %ss", path.name, ev.get("seconds"))
        return path

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - best effort teardown
            self._proc.kill()

    # ---------------------------------------------------------------------- access
    @classmethod
    def get(cls) -> AceStepWorker:
        with cls._lock:
            if cls._instance is None or cls._instance._proc.poll() is not None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None


def is_available() -> bool:
    """True if ACE-Step is installed. Does not spawn anything."""
    return acestep_python() is not None
