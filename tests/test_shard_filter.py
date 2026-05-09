"""Hash-mod shard filter test (Component 1).

The streaming sampler runs ``num_shards`` daemons in parallel (one per GPU).
Work distribution is purely static: each shard processes input lines whose
``line_index % num_shards == shard_id``. No coordinator, no claims.

Tests:
* shard 1 of 4 over 100 lines processes exactly indices {1, 5, 9, ..., 97}.
* shard 0 of 1 takes everything.
* shard >= num_shards errors out (CLI validation).
"""

from __future__ import annotations

import pytest

from sample.sample_northstar import iter_shard


def _items(n: int) -> list[dict]:
    return [{"item_id": str(i), "instruction": f"x{i}"} for i in range(n)]


def test_shard_1_of_4_picks_correct_indices():
    items = _items(100)
    out = list(iter_shard(items, shard=1, num_shards=4))
    expected = list(range(1, 100, 4))  # 1, 5, 9, ..., 97
    assert [int(it["item_id"]) for it in out] == expected
    assert len(out) == 25


def test_shard_0_of_1_takes_everything():
    items = _items(50)
    out = list(iter_shard(items, shard=0, num_shards=1))
    assert len(out) == 50
    assert [int(it["item_id"]) for it in out] == list(range(50))


def test_shard_id_out_of_range_raises():
    items = _items(10)
    with pytest.raises(ValueError):
        # shard 5 of 4 is invalid (valid: 0..3)
        list(iter_shard(items, shard=5, num_shards=4))


def test_negative_shard_raises():
    items = _items(10)
    with pytest.raises(ValueError):
        list(iter_shard(items, shard=-1, num_shards=4))


def test_zero_num_shards_raises():
    items = _items(10)
    with pytest.raises(ValueError):
        list(iter_shard(items, shard=0, num_shards=0))


def test_all_shards_partition_input_exactly():
    """Union of all shards == input; intersection == empty."""
    n = 97  # prime: catches off-by-one in modulo
    items = _items(n)
    seen: list[int] = []
    for s in range(4):
        out = list(iter_shard(items, shard=s, num_shards=4))
        for it in out:
            seen.append(int(it["item_id"]))
    assert sorted(seen) == list(range(n))
