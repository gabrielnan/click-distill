"""One-shot streaming-RSSD shard dashboard.

Reads ``<samples-root>/shard_*/status.json`` (written by
``StreamingWriter._write_status_snapshot`` after every item) plus a
``nvidia-smi`` snapshot, and prints the C-level dashboard from the grilling
session.

Layout (mirrors the spec):
    shards: 4/4 alive (last write 8s ago)
    items_done: 9234 (rate: 7.2/sec, +432/min)
    sft_items: 6012 (yield: 65.1%)
    elapsed: 22min
    projected at T+45min: ~18000 items, ~11700 SFT
    GPUs: gpu0=84% gpu1=81% gpu2=88% gpu3=82%
    --------------------------------------------------------------
    shard_0: 2308 done, 1521 sft (last write 7s ago)  alive
    shard_1: 2295 done, 1487 sft (last write 12s ago) alive
    shard_2: 2316 done, 1502 sft (last write 5s ago)  alive
    shard_3: 2315 done, 1502 sft (last write 9s ago)  alive

Usage:
    python sample/status.py                          # default --samples-root samples
    python sample/status.py --samples-root /path/to/samples
    python sample/status.py --target-min 45          # change projection horizon
    python sample/status.py --watch 5                # refresh every 5s

Pure functions are unit-tested in tests/test_status.py; the CLI wrapper at the
bottom is intentionally thin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ANSI red — use only when stdout is a tty.
_RED = "\033[31m"
_RESET = "\033[0m"

STALE_THRESHOLD_SEC = 60


# ---------------------------------------------------------------------------
# Pure helpers (tested directly)
# ---------------------------------------------------------------------------
def load_shard_status(samples_root: Path) -> list[dict[str, Any]]:
    """Read every ``shard_*/status.json`` under ``samples_root``.

    Returns a list of dicts (sorted by ``shard_id``), each augmented with
    ``last_write_age_sec`` (seconds since last write per the writer's clock).
    Shards whose status.json is missing or malformed are silently skipped —
    callers should assume the shard hasn't started yet.
    """
    out: list[dict[str, Any]] = []
    if not samples_root.exists():
        return out
    now = time.time()
    for shard_dir in sorted(samples_root.glob("shard_*")):
        status_path = shard_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            data = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["last_write_age_sec"] = max(0.0, now - float(data.get("last_write_at", now)))
        data["elapsed_sec"] = max(0.0, now - float(data.get("started_at", now)))
        out.append(data)
    out.sort(key=lambda s: s.get("shard_id", 0))
    return out


def is_stale(status: dict[str, Any], *, threshold_sec: float = STALE_THRESHOLD_SEC) -> bool:
    """A shard is stale if its last write is older than ``threshold_sec``.

    The writer flushes status.json after every item, so a healthy shard
    refreshes well within 60s even at modest throughput.
    """
    last = status.get("last_write_at")
    if last is None:
        return True
    return (time.time() - float(last)) > threshold_sec


def aggregate_status(
    statuses: list[dict[str, Any]],
    *,
    stale_threshold_sec: float = STALE_THRESHOLD_SEC,
) -> dict[str, Any]:
    """Sum the per-shard counters into a single dict for the dashboard header.

    Reports:
      * ``num_shards``  — how many status.json files we found
      * ``alive``       — how many are NOT stale
      * ``items_done``, ``sft_items``, ``zero_hit_items`` — sums
      * ``yield_pct``   — sft / done
      * ``elapsed_sec`` — max across shards (== wall time of the daemon)
      * ``rate_per_sec``— items_done / elapsed_sec
      * ``min_last_write_age_sec`` — most recent activity across the whole
        cluster (the "last write Xs ago" in the header)
    """
    items_done = sum(int(s.get("items_done", 0)) for s in statuses)
    sft_items = sum(int(s.get("sft_items", 0)) for s in statuses)
    zero = sum(int(s.get("zero_hit_items", 0)) for s in statuses)
    elapsed = max((float(s.get("elapsed_sec", 0.0)) for s in statuses), default=0.0)
    alive = sum(1 for s in statuses if not is_stale(s, threshold_sec=stale_threshold_sec))
    min_age = min(
        (float(s.get("last_write_age_sec", float("inf"))) for s in statuses),
        default=float("inf"),
    )
    return {
        "num_shards": len(statuses),
        "alive": alive,
        "items_done": items_done,
        "sft_items": sft_items,
        "zero_hit_items": zero,
        "yield_pct": (sft_items / items_done * 100.0) if items_done else 0.0,
        "elapsed_sec": elapsed,
        "rate_per_sec": (items_done / elapsed) if elapsed > 0 else 0.0,
        "min_last_write_age_sec": min_age,
    }


def project_at(
    *,
    rate_per_sec: float,
    yield_pct: float,
    elapsed_sec: float,
    target_total_sec: float,
) -> dict[str, Any]:
    """Linear extrapolation of items_done / sft at a future wall-clock time.

    ``target_total_sec`` is the absolute elapsed time (e.g. 45min from start),
    NOT a delta. Returns zeros if we're already past the target.
    """
    remaining = max(0.0, target_total_sec - elapsed_sec)
    # We don't know the current count here — caller passes rate; multiply by
    # full target_total_sec to get an "asymptote" projection (the spec's
    # "projected at T+45min: ~18000 items"). This treats the rate as steady
    # from t=0, which matches the dashboard wording.
    projected_items = int(rate_per_sec * target_total_sec)
    projected_sft = int(projected_items * yield_pct / 100.0)
    return {
        "projected_items_done": projected_items,
        "projected_sft_items": projected_sft,
        "remaining_sec": remaining,
    }


# ---------------------------------------------------------------------------
# GPU util helper
# ---------------------------------------------------------------------------
def gpu_utils() -> dict[str, str]:
    """Map ``gpu_id_str -> util_pct_str`` via ``nvidia-smi``.

    Returns ``{}`` (treated as N/A by the formatter) if nvidia-smi is missing
    or fails — keeps the dashboard usable on a Mac for testing.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    util_by_id: dict[str, str] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            util_by_id[parts[0]] = parts[1]
    return util_by_id


# ---------------------------------------------------------------------------
# Dashboard formatter
# ---------------------------------------------------------------------------
def _fmt_age(secs: float) -> str:
    if secs == float("inf"):
        return "?"
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}min"
    return f"{secs / 3600:.1f}h"


def format_dashboard(
    statuses: list[dict[str, Any]],
    *,
    gpu_utils: dict[str, str] | None = None,
    target_min: int = 45,
    use_color: bool = False,
) -> str:
    """Render the dashboard into a single string. Pure — no I/O, no clock
    reads beyond what's already encoded in ``statuses`` (loader populates
    ``last_write_age_sec``)."""
    gpu_utils = gpu_utils or {}

    if not statuses:
        return (
            "shards: 0/0 alive (no shards found yet)\n"
            "(launch with sample/launch_shards.sh)"
        )

    agg = aggregate_status(statuses)
    elapsed_min = agg["elapsed_sec"] / 60.0
    rate_per_min = agg["rate_per_sec"] * 60.0

    proj = project_at(
        rate_per_sec=agg["rate_per_sec"],
        yield_pct=agg["yield_pct"],
        elapsed_sec=agg["elapsed_sec"],
        target_total_sec=target_min * 60,
    )

    lines: list[str] = []
    lines.append(
        f"shards: {agg['alive']}/{agg['num_shards']} alive "
        f"(last write {_fmt_age(agg['min_last_write_age_sec'])} ago)"
    )
    lines.append(
        f"items_done: {agg['items_done']} "
        f"(rate: {agg['rate_per_sec']:.1f}/sec, +{rate_per_min:.0f}/min)"
    )
    lines.append(
        f"sft_items: {agg['sft_items']} (yield: {agg['yield_pct']:.1f}%)"
    )
    lines.append(f"elapsed: {elapsed_min:.0f}min")
    lines.append(
        f"projected at T+{target_min}min: "
        f"~{proj['projected_items_done']} items, "
        f"~{proj['projected_sft_items']} SFT"
    )
    if gpu_utils:
        gpu_str = " ".join(
            f"gpu{gid}={u}%" for gid, u in sorted(gpu_utils.items())
        )
        lines.append(f"GPUs: {gpu_str}")
    else:
        lines.append("GPUs: N/A")
    lines.append("-" * 62)

    for s in statuses:
        sid = s.get("shard_id", "?")
        done = s.get("items_done", 0)
        sft = s.get("sft_items", 0)
        age = s.get("last_write_age_sec", float("inf"))
        stale = is_stale(s)
        marker = "STALE" if stale else "alive"
        if stale and use_color:
            marker = f"{_RED}STALE{_RESET}"
        lines.append(
            f"shard_{sid}: {done} done, {sft} sft "
            f"(last write {_fmt_age(age)} ago)  {marker}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------
def _print_once(samples_root: Path, target_min: int, use_color: bool) -> None:
    statuses = load_shard_status(samples_root)
    print(format_dashboard(
        statuses,
        gpu_utils=gpu_utils(),
        target_min=target_min,
        use_color=use_color,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-root", type=Path, default=Path("samples"),
        help="Root containing shard_<i>/ subdirs (default: samples)",
    )
    parser.add_argument(
        "--target-min", type=int, default=45,
        help="Projection horizon (default: 45min)",
    )
    parser.add_argument(
        "--watch", type=float, default=None,
        help="Refresh every N seconds. Default: print once and exit.",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour for STALE markers (default: enabled if TTY).",
    )
    args = parser.parse_args()

    use_color = sys.stdout.isatty() and not args.no_color

    if args.watch is None:
        _print_once(args.samples_root, args.target_min, use_color)
        return 0

    try:
        while True:
            # Cheap clear: just reprint with a separator. Avoids OS-specific
            # escape sequences and lets the output be piped to less etc.
            print("=" * 62)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            _print_once(args.samples_root, args.target_min, use_color)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
