"""Wall-clock timing per stage, appended to timings.jsonl.

Stage durations decide the talk's running order, so these need to be real measurements
rather than estimates.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from .config import TIMINGS


@contextmanager
def stage(name: str, **meta):
    """Time a block and append one JSON line. Records duration even when the block raises."""
    t0 = time.perf_counter()
    error = None
    try:
        yield meta
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record = {
            "stage": name,
            "seconds": round(time.perf_counter() - t0, 3),
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            **meta,
        }
        if error:
            record["error"] = error
        TIMINGS.parent.mkdir(parents=True, exist_ok=True)
        with TIMINGS.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


def read_timings() -> list[dict]:
    if not TIMINGS.exists():
        return []
    out = []
    for line in TIMINGS.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def summary() -> str:
    """Per-stage table: count, total, mean, slowest. This is the pacing tool."""
    rows = read_timings()
    if not rows:
        return "No timings recorded yet."

    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["stage"], []).append(r["seconds"])

    width = max(len(s) for s in by) + 2
    lines = [f"{'stage':<{width}}{'n':>4}{'total':>10}{'mean':>10}{'max':>10}", "-" * (width + 34)]
    for name, secs in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        lines.append(
            f"{name:<{width}}{len(secs):>4}{sum(secs):>10.2f}"
            f"{sum(secs) / len(secs):>10.2f}{max(secs):>10.2f}"
        )
    lines.append("-" * (width + 34))
    lines.append(f"{'TOTAL':<{width}}{len(rows):>4}{sum(sum(v) for v in by.values()):>10.2f}")
    return "\n".join(lines)
