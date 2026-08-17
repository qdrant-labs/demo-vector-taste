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
import re
import subprocess
import threading
import time
from pathlib import Path

from .config import ROOT

log = logging.getLogger("vector_taste.worker")

SENTINEL = "@@VT@@"
WORKER = ROOT / "scripts" / "acestep_worker.py"

# tqdm line from the MLX diffusion loop, e.g.
#   "MLX DiT diffusion:  25%|██▌       | 2/8 [01:02<02:49, 28.23s/it]"
# This is the ONLY per-step signal available: ACE-Step's own callback jumps straight from
# 0.52 to 0.80 across the whole diffusion phase.
_TQDM = re.compile(r"diffusion:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)", re.IGNORECASE)

# ACE-Step's raw fractions remapped to 0-1 in proportion to MEASURED wall clock. Its own
# scale would start the bar at 51% (the LM phases never fire for us) and would sit frozen
# through diffusion, which is 69s of a ~102s run.
_DIFFUSION_START = 0.05
_DIFFUSION_END = 0.75
_DECODE_END = 0.98


def parse_diffusion_step(line: str) -> tuple[int, int] | None:
    """(step, total) from a tqdm diffusion line, or None if it isn't one."""
    m = _TQDM.search(line)
    if not m:
        return None
    step, total = int(m.group(1)), int(m.group(2))
    return (step, total) if total > 0 else None


def map_fraction(raw: float) -> tuple[float, str]:
    """ACE-Step's fraction -> (display fraction, phase label)."""
    if raw >= 0.99:
        return _DECODE_END, "writing audio"
    if raw >= 0.80:
        return _DIFFUSION_END, "decoding audio"
    if raw >= 0.52:
        return _DIFFUSION_START, "diffusion"
    return 0.0, "preparing"


class Progress:
    """Latest progress for the in-flight job. One job runs at a time (job lock)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self, phase: str = "idle") -> None:
        with self._lock:
            self._d = {
                "phase": phase, "frac": 0.0, "step": 0, "total": 0,
                "desc": "", "started": time.time(), "job": None,
            }

    def update(self, **kw) -> None:
        with self._lock:
            # Never let the bar go backwards: tqdm and the callback interleave, and a
            # late-arriving lower fraction would look like a stall or a restart.
            if "frac" in kw and kw["frac"] < self._d.get("frac", 0.0):
                kw.pop("frac")
            self._d.update(kw)

    def snapshot(self) -> dict:
        with self._lock:
            d = dict(self._d)
        d["elapsed"] = round(time.time() - d.pop("started", time.time()), 1)
        return d


PROGRESS = Progress()

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
            # PIPE, not DEVNULL: the diffusion tqdm bar is the only per-step progress
            # signal there is. It MUST be drained continuously -- an unread pipe fills its
            # ~64KB buffer and blocks the worker mid-generation, which presents as a hang
            # with no error at all. `_drain_stderr` below is that drain.
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        self._job_lock = threading.Lock()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        log.info("ACE-Step worker starting (this takes a minute or two on first run)")
        PROGRESS.reset("loading model")
        ready = self._read_event(READY_TIMEOUT)
        if not ready.get("ok"):
            raise WorkerError(ready.get("error") or "worker failed to start")
        PROGRESS.reset("idle")
        log.info("ACE-Step worker ready in %ss", ready.get("seconds"))

    def _drain_stderr(self) -> None:
        """Consume the worker's stderr forever, parsing diffusion progress out of it.

        Two jobs, and the first is the important one: keep the pipe empty. tqdm rewrites its
        bar with \\r rather than \\n, so read in small chunks instead of by line -- iterating
        lines would buffer the entire diffusion phase into one unhelpful blob.
        """
        # os.read on the raw fd, NOT stderr.read(n): a sized read on the text stream blocks
        # until n characters arrive, which batches several tqdm updates together and makes
        # the ring jump instead of tick. os.read returns as soon as anything is available.
        fd = self._proc.stderr.fileno()
        buf = ""
        try:
            while True:
                raw = os.read(fd, 4096)
                if not raw:
                    return
                buf += raw.decode("utf-8", "replace")
                # \r is tqdm's update separator; \n ends normal log lines.
                parts = re.split(r"[\r\n]", buf)
                buf = parts.pop()
                for part in parts:
                    got = parse_diffusion_step(part)
                    if not got:
                        continue
                    step, total = got
                    span = _DIFFUSION_END - _DIFFUSION_START
                    PROGRESS.update(
                        phase="diffusion",
                        step=step,
                        total=total,
                        frac=_DIFFUSION_START + span * (step / total),
                        desc=f"diffusion {step}/{total}",
                    )
        except Exception:  # noqa: BLE001 - draining must never take the worker down
            log.debug("stderr drain ended", exc_info=True)

    # -------------------------------------------------------------------- protocol
    def _read_event(self, timeout: float) -> dict:
        """Read until the next terminal sentinel line, applying progress events on the way.

        `progress` events are consumed and folded into PROGRESS rather than returned, so a
        caller waiting for a result is not woken by them.
        """
        result: dict = {}
        done = threading.Event()

        def reader():
            nonlocal result
            for line in self._proc.stdout:
                if not line.startswith(SENTINEL):
                    continue
                try:
                    ev = json.loads(line[len(SENTINEL):].strip())
                except json.JSONDecodeError:
                    continue
                if ev.get("event") == "progress":
                    frac, phase = map_fraction(ev.get("frac", 0.0))
                    # Don't let a coarse callback stomp the finer tqdm step count.
                    update = {"phase": phase, "frac": frac, "desc": ev.get("desc") or phase}
                    if phase != "diffusion":
                        update["step"] = update["total"] = 0
                    PROGRESS.update(**update)
                    continue
                result = ev
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
                PROGRESS.reset("idle")
                raise WorkerError("ACE-Step worker has exited; it will respawn next call")
            PROGRESS.reset("preparing")
            PROGRESS.update(job=job.get("id"), desc="preparing")
            self._proc.stdin.write(json.dumps(job) + "\n")
            self._proc.stdin.flush()
            try:
                ev = self._read_event(JOB_TIMEOUT)
            finally:
                PROGRESS.update(phase="done", frac=1.0, desc="done")

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


_warming = threading.Event()


def worker_state() -> str:
    """unavailable | idle | warming | ready — for /api/status, spawns nothing."""
    if not is_available():
        return "unavailable"
    inst = AceStepWorker._instance
    if inst is not None and inst._proc.poll() is None:
        return "ready"
    return "warming" if _warming.is_set() else "idle"


def prewarm() -> None:
    """Load the model in the background so the first compose isn't a 150s blank wait.

    Non-blocking by design: the UI must serve search immediately. Failures are logged and
    swallowed — a machine without ACE-Step should still get a working search demo, and the
    real error surfaces when someone actually composes.
    """
    if not is_available() or _warming.is_set():
        return

    def run():
        _warming.set()
        try:
            AceStepWorker.get()
        except Exception as exc:  # noqa: BLE001
            log.warning("pre-warm failed (generation will retry on demand): %s", exc)
        finally:
            _warming.clear()

    threading.Thread(target=run, daemon=True).start()
