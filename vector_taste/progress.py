"""Generation progress and cancellation, shared across backends.

This used to live inside `worker.py`, coupled to the local ACE-Step subprocess: abort meant
"kill the process". A hosted backend has no process to kill, so Stop would have silently
done nothing on it — worse than not offering a Stop button at all.

So both concerns move here and become backend-agnostic:

  PROGRESS   one snapshot of whatever is generating, polled by the UI ring
  aborting   a registry -- whichever backend is running registers how to cancel itself
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger("vector_taste.progress")


class Progress:
    """Latest progress for the in-flight job. One job runs at a time (job lock)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self, phase: str = "idle") -> None:
        with self._lock:
            self._d = {
                "phase": phase, "frac": 0.0, "step": 0, "total": 0,
                "desc": "", "started": time.time(), "job": None, "backend": "",
            }

    def update(self, **kw) -> None:
        with self._lock:
            # Never let the bar go backwards: sources interleave (ACE-Step's callback and
            # its diffusion tqdm), and a late-arriving lower fraction would read as a stall.
            if "frac" in kw and kw["frac"] < self._d.get("frac", 0.0):
                kw.pop("frac")
            self._d.update(kw)

    def snapshot(self) -> dict:
        with self._lock:
            d = dict(self._d)
        d["elapsed"] = round(time.time() - d.pop("started", time.time()), 1)
        return d


PROGRESS = Progress()


# --------------------------------------------------------------------------- aborting
_abort_lock = threading.Lock()
_aborter: Callable[[], bool] | None = None


def register_aborter(fn: Callable[[], bool] | None) -> None:
    """Declare how to cancel the currently running generation.

    Each backend knows its own cancellation: ACE-Step kills its worker process, a hosted
    backend closes its HTTP client. Registering `None` clears it, so Stop after a run has
    finished reports "nothing was generating" rather than cancelling the next one.
    """
    global _aborter
    with _abort_lock:
        _aborter = fn


def abort_current() -> bool:
    """Cancel the running generation. False if nothing was running.

    Never raises: Stop must not be able to fail in front of an audience.
    """
    with _abort_lock:
        fn = _aborter
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - a failed cancel must not surface as a crash
        log.exception("aborter raised")
        return False


def is_aborting() -> bool:
    with _abort_lock:
        return _aborter is not None
