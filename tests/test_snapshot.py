"""Snapshot freeze test (Component 2).

Streaming sampler writes to ``samples/shard_{0..N-1}/sft.jsonl``. Before each
training run we freeze the current state of all shards into
``snapshots/run_<id>/sft.jsonl`` (concatenated) plus a tiny
``snapshot_meta.json`` for provenance. Sampling continues writing to the
source ``samples/`` dir uninterrupted.

Invariants under test:
* Concatenated ``sft.jsonl`` contains the union of every shard's lines.
* ``snapshot_meta.json`` records run_id, created_at, source_shards, total_sft_items.
* Source directory and source files are untouched (no mutation, no rename).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from train.freeze_snapshot import freeze_snapshot


def _write_shard(samples_dir: Path, shard_id: int, lines: list[dict]) -> None:
    sd = samples_dir / f"shard_{shard_id}"
    sd.mkdir(parents=True, exist_ok=True)
    with (sd / "sft.jsonl").open("w", encoding="utf-8") as fh:
        for r in lines:
            fh.write(json.dumps(r) + "\n")
    # Status file analogous to what the sampler writes.
    (sd / "status.json").write_text(json.dumps({
        "shard_id": shard_id,
        "items_done": len(lines),
        "sft_items": len(lines),
    }))


def test_freeze_snapshot_creates_concatenated_jsonl(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    snapshots_dir = tmp_path / "snapshots"
    _write_shard(samples_dir, 0, [{"item_id": "0_a"}, {"item_id": "0_b"}])
    _write_shard(samples_dir, 1, [{"item_id": "1_a"}])
    _write_shard(samples_dir, 2, [])
    _write_shard(samples_dir, 3, [{"item_id": "3_a"}, {"item_id": "3_b"}, {"item_id": "3_c"}])

    snap = freeze_snapshot(samples_dir, snapshots_dir, run_id="1130")

    assert snap.is_dir()
    assert snap == snapshots_dir / "run_1130"

    sft = snap / "sft.jsonl"
    assert sft.is_file()
    rows = [json.loads(l) for l in sft.read_text().splitlines() if l.strip()]
    ids = sorted(r["item_id"] for r in rows)
    assert ids == ["0_a", "0_b", "1_a", "3_a", "3_b", "3_c"]


def test_freeze_snapshot_writes_meta(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    snapshots_dir = tmp_path / "snapshots"
    _write_shard(samples_dir, 0, [{"item_id": "x"}])
    _write_shard(samples_dir, 1, [{"item_id": "y"}, {"item_id": "z"}])

    t_before = time.time()
    snap = freeze_snapshot(samples_dir, snapshots_dir, run_id="1200")
    t_after = time.time()

    meta_path = snap / "snapshot_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())

    assert meta["run_id"] == "1200"
    assert meta["total_sft_items"] == 3
    assert sorted(meta["source_shards"]) == ["shard_0", "shard_1"]
    assert t_before <= meta["created_at"] <= t_after


def test_freeze_snapshot_does_not_mutate_source(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    snapshots_dir = tmp_path / "snapshots"
    rows = [{"item_id": "0_a"}, {"item_id": "0_b"}]
    _write_shard(samples_dir, 0, rows)
    src_path = samples_dir / "shard_0" / "sft.jsonl"
    src_text_before = src_path.read_text()
    src_status_before = (samples_dir / "shard_0" / "status.json").read_text()

    freeze_snapshot(samples_dir, snapshots_dir, run_id="9999")

    assert src_path.read_text() == src_text_before, "source sft.jsonl was mutated"
    assert (samples_dir / "shard_0" / "status.json").read_text() == src_status_before
    # No new files in the source dir.
    files = sorted(p.name for p in (samples_dir / "shard_0").iterdir())
    assert files == ["sft.jsonl", "status.json"]


def test_freeze_snapshot_handles_missing_sft_jsonl(tmp_path: Path):
    """A shard that has a status.json but no sft.jsonl yet (sampler still
    spinning up) should be tolerated, contributing zero rows."""
    samples_dir = tmp_path / "samples"
    snapshots_dir = tmp_path / "snapshots"
    sd = samples_dir / "shard_0"
    sd.mkdir(parents=True)
    (sd / "status.json").write_text(json.dumps({"shard_id": 0, "items_done": 0}))
    _write_shard(samples_dir, 1, [{"item_id": "1_a"}])

    snap = freeze_snapshot(samples_dir, snapshots_dir, run_id="0000")
    rows = [json.loads(l) for l in (snap / "sft.jsonl").read_text().splitlines() if l.strip()]
    assert [r["item_id"] for r in rows] == ["1_a"]
    meta = json.loads((snap / "snapshot_meta.json").read_text())
    assert meta["total_sft_items"] == 1
    assert sorted(meta["source_shards"]) == ["shard_0", "shard_1"]


def test_freeze_snapshot_empty_samples_dir_raises(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    snapshots_dir = tmp_path / "snapshots"
    with pytest.raises(FileNotFoundError):
        freeze_snapshot(samples_dir, snapshots_dir, run_id="empty")


def test_freeze_snapshot_creates_target_dirs(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    snapshots_dir = tmp_path / "snapshots" / "deep" / "nested"
    _write_shard(samples_dir, 0, [{"item_id": "x"}])
    snap = freeze_snapshot(samples_dir, snapshots_dir, run_id="abc")
    assert snap.is_dir()
    assert (snap / "sft.jsonl").is_file()
