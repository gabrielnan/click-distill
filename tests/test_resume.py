"""Resume-on-restart test (Component 3).

The streaming sampler is a daemon: kill -9 → relaunch must pick up where it
left off without redoing work or skipping work. The resume contract:

  skip = read(done_ids.txt) ∪ read(zero_hits.txt)
  for item in input_stream:
      if hash_mod(item) != shard_id: continue
      if item.item_id in skip:        continue
      process(item)

The hash-mod and skip-set filters compose: resume narrows the per-shard
slice to "everything I haven't touched yet".
"""

from __future__ import annotations

from pathlib import Path

from sample.sample_northstar import (
    StreamingWriter,
    iter_jsonl,
    iter_shard,
    plan_shard_work,
)


def _items_jsonl(tmp_path: Path, n: int) -> Path:
    """Write a JSONL of ``n`` items with item_ids "0".."n-1" and return the path."""
    out = tmp_path / "input.jsonl"
    import json
    with out.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "item_id": str(i),
                "instruction": f"x{i}",
                "image_path": f"/fake/{i}.png",
                "bbox": [40, 40, 60, 60],
                "image_width": 1000,
                "image_height": 1000,
            }) + "\n")
    return out


def test_resume_skips_done_ids(tmp_path: Path):
    """Given existing done_ids.txt with [3, 7, 12, 22] and 100 input items,
    after resume the shard processes only items in its hash-mod slice MINUS
    those four ids."""
    input_path = _items_jsonl(tmp_path, 100)
    shard_dir = tmp_path / "shard_0"
    shard_dir.mkdir()

    # Simulate prior progress.
    (shard_dir / "done_ids.txt").write_text("3\n7\n12\n22\n")
    (shard_dir / "zero_hits.txt").write_text("")  # exists but empty

    writer = StreamingWriter(
        shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7,
    )
    skip = writer.load_skip_set()
    assert skip == {"3", "7", "12", "22"}

    # Plan the work for shard 0 of 1: full input minus the skip set.
    todo = list(plan_shard_work(
        input_path, shard=0, num_shards=1, skip_ids=skip,
    ))
    todo_ids = sorted(int(x["item_id"]) for x in todo)
    assert todo_ids == [i for i in range(100) if i not in {3, 7, 12, 22}]
    assert len(todo) == 96


def test_resume_combines_with_hash_mod(tmp_path: Path):
    """Hash-mod and skip-set compose: shard 1 of 4 starts with {1,5,9,...,97}.
    Mark 5 and 13 done — resume should yield {1, 9, 17, 21, ..., 97}."""
    input_path = _items_jsonl(tmp_path, 100)
    shard_dir = tmp_path / "shard_1"
    shard_dir.mkdir()
    (shard_dir / "done_ids.txt").write_text("5\n13\n")

    writer = StreamingWriter(
        shard_dir, shard_id=1, gpu_id="1", n_per_item=8, temperature=0.7,
    )
    skip = writer.load_skip_set()

    todo = list(plan_shard_work(
        input_path, shard=1, num_shards=4, skip_ids=skip,
    ))
    todo_ids = sorted(int(x["item_id"]) for x in todo)
    expected = [i for i in range(1, 100, 4) if i not in {5, 13}]
    assert todo_ids == expected


def test_resume_includes_zero_hits(tmp_path: Path):
    """Items in zero_hits.txt must NOT be re-tried (re-sampling won't help)."""
    input_path = _items_jsonl(tmp_path, 20)
    shard_dir = tmp_path / "shard_0"
    shard_dir.mkdir()
    (shard_dir / "done_ids.txt").write_text("4\n5\n")
    (shard_dir / "zero_hits.txt").write_text("4\n5\n")  # both were 0-hit

    writer = StreamingWriter(
        shard_dir, shard_id=0, gpu_id="0", n_per_item=8, temperature=0.7,
    )
    skip = writer.load_skip_set()
    assert {"4", "5"}.issubset(skip)

    todo_ids = sorted(int(x["item_id"]) for x in plan_shard_work(
        input_path, shard=0, num_shards=1, skip_ids=skip,
    ))
    assert 4 not in todo_ids
    assert 5 not in todo_ids


def test_resume_with_no_prior_state(tmp_path: Path):
    """Fresh start: empty skip set, full hash-mod slice processed."""
    input_path = _items_jsonl(tmp_path, 40)
    shard_dir = tmp_path / "shard_2"
    writer = StreamingWriter(
        shard_dir, shard_id=2, gpu_id="2", n_per_item=8, temperature=0.7,
    )
    skip = writer.load_skip_set()
    assert skip == set()
    todo = list(plan_shard_work(
        input_path, shard=2, num_shards=4, skip_ids=skip,
    ))
    assert sorted(int(x["item_id"]) for x in todo) == list(range(2, 40, 4))


def test_iter_jsonl_tags_line_index(tmp_path: Path):
    """iter_jsonl must set _line_index so hash-mod is deterministic regardless
    of how the item dict was constructed downstream."""
    input_path = _items_jsonl(tmp_path, 5)
    items = list(iter_jsonl(input_path))
    assert [it["_line_index"] for it in items] == [0, 1, 2, 3, 4]
