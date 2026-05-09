"""Streaming writer test (Component 2).

The writer owns 4 files inside ``samples/shard_<id>/``:
* ``sft.jsonl``      — append-only, one row per WINNING completion only
* ``done_ids.txt``   — append-only, one item_id per line (covers BOTH
                       items that produced an SFT row AND items that hit 0/8)
* ``zero_hits.txt``  — append-only, item_ids with 0/8 hits (subset of
                       done_ids; lets us tell yield apart from progress)
* ``status.json``    — atomic snapshot used by the status dashboard

Per-item ordering invariant (crash-safety):
  1. Append SFT row to sft.jsonl, flush + fsync.
  2. THEN append item_id to done_ids.txt, flush + fsync.

This ordering means a crash between (1) and (2) replays the SFT row on resume
(harmless — we'll dedupe by item_id anyway, and the SFT consumer doesn't care
about a dup), but a crash AFTER (2) is invisible to consumers — done_ids
never references an SFT row that isn't on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sample.sample_northstar import StreamingWriter


def _hit() -> str:
    return (
        '<tool_call>{"name":"computer_use","arguments":'
        '{"type":"click","x":50,"y":50}}</tool_call>'
    )


def _miss() -> str:
    return (
        '<tool_call>{"name":"computer_use","arguments":'
        '{"type":"click","x":900,"y":900}}</tool_call>'
    )


def _item(idx: int) -> dict:
    return {
        "item_id": str(idx),
        "instruction": f"click thing {idx}",
        "image_path": f"/fake/img_{idx}.png",
        "bbox": [40, 40, 60, 60],
        "image_width": 1000,
        "image_height": 1000,
    }


def test_writer_creates_layout(tmp_path: Path):
    """The writer creates the per-shard directory structure on construction."""
    shard_dir = tmp_path / "shard_0"
    StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)
    assert shard_dir.is_dir()
    assert (shard_dir / "status.json").exists()


def test_writer_writes_winning_completion_only(tmp_path: Path):
    """Per item with 8 completions, write exactly the first hit (or none)."""
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)

    item = _item(1)
    # 3 misses, then a hit, then more misses+hits — should keep the FIRST hit.
    completions = [_miss(), _miss(), _miss(), _hit(), _miss(), _hit(), _miss(), _hit()]
    n_hits = w.write_item(item, completions)
    assert n_hits == 3  # total hits across the 8 — for stats, not for output

    sft_lines = (shard_dir / "sft.jsonl").read_text().strip().splitlines()
    assert len(sft_lines) == 1, "should write exactly one SFT row (the first hit)"
    row = json.loads(sft_lines[0])
    assert row["item_id"] == "1"
    assert row["completion"] == completions[3]
    assert row["instruction"] == item["instruction"]
    assert row["image_path"] == item["image_path"]


def test_writer_records_zero_hit_items(tmp_path: Path):
    """An item with 0/8 hits: no SFT row, but it lands in zero_hits.txt + done_ids.txt."""
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)

    item = _item(7)
    completions = [_miss()] * 8
    n_hits = w.write_item(item, completions)
    assert n_hits == 0

    sft = shard_dir / "sft.jsonl"
    if sft.exists():
        assert sft.read_text() == ""  # nothing written for 0-hit items
    zero_hits = (shard_dir / "zero_hits.txt").read_text().strip().splitlines()
    assert zero_hits == ["7"]
    done = (shard_dir / "done_ids.txt").read_text().strip().splitlines()
    assert done == ["7"]


def test_writer_done_ids_tracks_all_items(tmp_path: Path):
    """5 items, 3 hit, 2 zero — sft.jsonl has 3 rows; done_ids has all 5; zero_hits has 2."""
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)

    # Items 1, 2, 3 hit; items 4, 5 miss everything.
    for idx in (1, 2, 3):
        w.write_item(_item(idx), [_miss(), _hit()] + [_miss()] * 6)
    for idx in (4, 5):
        w.write_item(_item(idx), [_miss()] * 8)

    sft_lines = (shard_dir / "sft.jsonl").read_text().strip().splitlines()
    assert len(sft_lines) == 3
    sft_ids = [json.loads(line)["item_id"] for line in sft_lines]
    assert sft_ids == ["1", "2", "3"]

    done = (shard_dir / "done_ids.txt").read_text().strip().splitlines()
    assert done == ["1", "2", "3", "4", "5"]

    zero_hits = (shard_dir / "zero_hits.txt").read_text().strip().splitlines()
    assert zero_hits == ["4", "5"]


def test_writer_calls_fsync_per_item(tmp_path: Path):
    """fsync should fire after EACH file write (sft, done_ids, zero_hits, status)."""
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)

    # Patch on the module that owns the writer.
    with patch("sample.sample_northstar.os.fsync") as mock_fsync:
        w.write_item(_item(1), [_hit()] * 8)  # hit -> sft + done + status
    # At minimum: sft.jsonl, done_ids.txt, status.json — 3 fsync calls.
    assert mock_fsync.call_count >= 3

    with patch("sample.sample_northstar.os.fsync") as mock_fsync:
        w.write_item(_item(2), [_miss()] * 8)  # zero_hit -> zero_hits + done + status
    assert mock_fsync.call_count >= 3


def test_done_ids_appended_after_sft(tmp_path: Path):
    """Crash-safety: simulate a crash IMMEDIATELY after sft write, BEFORE done_ids
    write. On resume done_ids must NOT reference the SFT row (it doesn't yet),
    so the worst case is a duplicate SFT row, never a missing one.
    """
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)

    crash_after_sft = {"crashed": False}
    real_open = open

    original_append_done = w._append_done_id

    def crashing_append(item_id):
        crash_after_sft["crashed"] = True
        raise IOError("simulated crash between sft.jsonl and done_ids.txt")

    w._append_done_id = crashing_append  # type: ignore[assignment]

    with pytest.raises(IOError):
        w.write_item(_item(42), [_hit()] * 8)

    # After "crash": sft.jsonl has the row, done_ids.txt is empty/missing.
    sft_lines = (shard_dir / "sft.jsonl").read_text().strip().splitlines()
    assert len(sft_lines) == 1
    assert json.loads(sft_lines[0])["item_id"] == "42"

    done_path = shard_dir / "done_ids.txt"
    if done_path.exists():
        # If it exists, it must NOT contain "42" (we crashed before append).
        assert "42" not in done_path.read_text().strip().splitlines()


def test_status_json_updates_on_each_write(tmp_path: Path):
    """status.json carries running counters used by the dashboard."""
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(
        shard_dir, shard_id=0, gpu_id="3", n_per_item=8, temperature=0.7,
    )
    status0 = json.loads((shard_dir / "status.json").read_text())
    assert status0["shard_id"] == 0
    assert status0["gpu_id"] == "3"
    assert status0["n_per_item"] == 8
    assert status0["temperature"] == 0.7
    assert status0["items_done"] == 0
    assert status0["sft_items"] == 0
    assert "started_at" in status0

    w.write_item(_item(1), [_hit()] + [_miss()] * 7)  # 1 hit -> sft row
    w.write_item(_item(2), [_miss()] * 8)              # zero hit -> no sft row

    status1 = json.loads((shard_dir / "status.json").read_text())
    assert status1["items_done"] == 2
    assert status1["sft_items"] == 1
    assert status1["zero_hit_items"] == 1
    assert status1["last_write_at"] >= status0["started_at"]


def test_writer_keep_first_short_circuits(tmp_path: Path):
    """``--keep first`` writes exactly the first hit and reports that as the win,
    even if later samples in the batch would also have hit."""
    shard_dir = tmp_path / "shard_0"
    w = StreamingWriter(shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7)
    completions = [_miss(), _hit(), _hit(), _hit()]
    w.write_item(_item(1), completions)
    sft = (shard_dir / "sft.jsonl").read_text().strip().splitlines()
    assert len(sft) == 1
    assert json.loads(sft[0])["completion"] == completions[1]
