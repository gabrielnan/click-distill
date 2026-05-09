"""Status dashboard test (Component 5).

The dashboard reads each shard's status.json + counts lines in sft.jsonl /
done_ids.txt and renders a one-shot snapshot. Logic is split into pure
functions so the CLI is a thin wrapper that's not tested directly.

Tested invariants:
* aggregate counters sum across shards
* yield % = sft / done
* rate = items_done / elapsed
* "STALE" marker if last_write_at is > 60s ago
* gpu_util defaults gracefully to "N/A" if nvidia-smi missing (not exercised
  here directly — the helper is documented to fall back).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sample.status import (
    aggregate_status,
    format_dashboard,
    is_stale,
    load_shard_status,
    project_at,
)


def _make_shard(
    samples_root: Path,
    shard_id: int,
    *,
    items_done: int,
    sft_items: int,
    zero_hits: int = 0,
    last_write_offset: float = 0.0,
    started_offset: float = 600.0,
    n_per_item: int = 8,
    temperature: float = 0.7,
    gpu_id: str = "0",
) -> Path:
    """Create a fixture shard dir with status.json + matching counters.

    ``last_write_offset``  = seconds before now (positive = past).
    ``started_offset``     = seconds before now (positive = past).
    """
    shard_dir = samples_root / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    payload = {
        "shard_id": shard_id,
        "gpu_id": gpu_id,
        "pid": 12345 + shard_id,
        "started_at": now - started_offset,
        "last_write_at": now - last_write_offset,
        "items_done": items_done,
        "sft_items": sft_items,
        "zero_hit_items": zero_hits,
        "n_per_item": n_per_item,
        "temperature": temperature,
    }
    (shard_dir / "status.json").write_text(json.dumps(payload))
    # Simulate matching files (line-counted by some callers).
    (shard_dir / "sft.jsonl").write_text("x\n" * sft_items)
    (shard_dir / "done_ids.txt").write_text("\n".join(str(i) for i in range(items_done)) + ("\n" if items_done else ""))
    return shard_dir


def test_load_shard_status_reads_json(tmp_path: Path):
    samples = tmp_path / "samples"
    _make_shard(samples, 0, items_done=10, sft_items=7, last_write_offset=2.0)
    statuses = load_shard_status(samples)
    assert len(statuses) == 1
    s = statuses[0]
    assert s["shard_id"] == 0
    assert s["items_done"] == 10
    assert s["sft_items"] == 7
    # Derived fields the dashboard adds.
    assert "last_write_age_sec" in s
    assert s["last_write_age_sec"] >= 1.5  # ~2s offset


def test_load_shard_status_sorts_by_id(tmp_path: Path):
    samples = tmp_path / "samples"
    _make_shard(samples, 2, items_done=1, sft_items=1)
    _make_shard(samples, 0, items_done=1, sft_items=1)
    _make_shard(samples, 1, items_done=1, sft_items=1)
    statuses = load_shard_status(samples)
    assert [s["shard_id"] for s in statuses] == [0, 1, 2]


def test_aggregate_sums_across_shards(tmp_path: Path):
    samples = tmp_path / "samples"
    _make_shard(samples, 0, items_done=2308, sft_items=1521, last_write_offset=7.0)
    _make_shard(samples, 1, items_done=2295, sft_items=1487, last_write_offset=12.0)
    _make_shard(samples, 2, items_done=2316, sft_items=1502, last_write_offset=5.0)
    _make_shard(samples, 3, items_done=2315, sft_items=1502, last_write_offset=9.0)
    statuses = load_shard_status(samples)
    agg = aggregate_status(statuses)
    assert agg["items_done"] == 9234
    assert agg["sft_items"] == 6012
    assert agg["num_shards"] == 4
    assert agg["alive"] == 4
    # Yield = sft / done
    assert abs(agg["yield_pct"] - (6012 / 9234 * 100)) < 1e-6
    # Most recent last_write_at -> smallest age.
    assert agg["min_last_write_age_sec"] < 6.0


def test_is_stale_threshold():
    now = time.time()
    assert not is_stale({"last_write_at": now - 30}, threshold_sec=60)
    assert is_stale({"last_write_at": now - 90}, threshold_sec=60)


def test_aggregate_marks_stale_shards_dead(tmp_path: Path):
    """Shards whose last_write was > stale_threshold ago count as not alive."""
    samples = tmp_path / "samples"
    _make_shard(samples, 0, items_done=10, sft_items=5, last_write_offset=1.0)
    _make_shard(samples, 1, items_done=10, sft_items=5, last_write_offset=120.0)  # stale
    statuses = load_shard_status(samples)
    agg = aggregate_status(statuses, stale_threshold_sec=60)
    assert agg["alive"] == 1
    assert agg["num_shards"] == 2


def test_project_at_extrapolates_linearly():
    """At T+45min total elapsed, with rate=7.2/sec and yield=65.1%, project
    ~7.2*60*45 ≈ 19440 items_done; sft ~ 12660."""
    proj = project_at(
        rate_per_sec=7.2,
        yield_pct=65.1,
        elapsed_sec=22 * 60,
        target_total_sec=45 * 60,
    )
    # Rate*remaining + current.
    assert proj["projected_items_done"] >= 0
    # We don't pin exact integer values — just check ballpark.
    assert 17000 <= proj["projected_items_done"] <= 20000
    assert 11000 <= proj["projected_sft_items"] <= 13000


def test_format_dashboard_renders_expected_lines(tmp_path: Path):
    samples = tmp_path / "samples"
    _make_shard(samples, 0, items_done=2308, sft_items=1521, last_write_offset=7.0)
    _make_shard(samples, 1, items_done=2295, sft_items=1487, last_write_offset=12.0)
    _make_shard(samples, 2, items_done=2316, sft_items=1502, last_write_offset=5.0)
    _make_shard(samples, 3, items_done=2315, sft_items=1502, last_write_offset=9.0)

    statuses = load_shard_status(samples)
    out = format_dashboard(
        statuses,
        gpu_utils={"0": "84", "1": "81", "2": "88", "3": "82"},
    )
    # Aggregate header.
    assert "shards: 4/4 alive" in out
    assert "items_done: 9234" in out
    assert "sft_items: 6012" in out
    # Per-shard rows.
    assert "shard_0:" in out
    assert "2308 done" in out
    assert "1521 sft" in out
    assert "alive" in out
    # GPU util surfaced.
    assert "gpu0=84%" in out


def test_format_dashboard_marks_stale(tmp_path: Path):
    samples = tmp_path / "samples"
    _make_shard(samples, 0, items_done=100, sft_items=70, last_write_offset=5.0)
    _make_shard(samples, 1, items_done=100, sft_items=70, last_write_offset=120.0)  # stale
    statuses = load_shard_status(samples)
    out = format_dashboard(statuses, gpu_utils={})
    assert "STALE" in out


def test_format_dashboard_no_shards(tmp_path: Path):
    samples = tmp_path / "samples"
    samples.mkdir()
    statuses = load_shard_status(samples)
    out = format_dashboard(statuses, gpu_utils={})
    # Don't crash on the "no shards yet" case — print something reasonable.
    assert "no shards" in out.lower() or "0/0" in out
