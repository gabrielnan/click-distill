"""Stratified subset sampler test (Component 1).

The intermediate eval needs a SMALL but REPRESENTATIVE slice of ScreenSpot-Pro
so we can run it ~10x during the hackathon (one per LoRA candidate). The full
1581-item eval is reserved for the best-looking adapter.

Constraints:
* Subset size = 200, stratified across 5 categories (~40 per category).
* SAME items across all training runs (else metric deltas mean nothing).
* Deterministic given a seed.
* Persistent on disk so a fresh process picks up the same subset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.stratified_subset import make_subset, load_subset


def _mock_items(per_cat: int = 300, cats=("Dev", "Creative", "CAD", "Scientific", "Office")) -> list[dict]:
    """1500 mock items, 300 in each of 5 categories."""
    items: list[dict] = []
    for cat in cats:
        for i in range(per_cat):
            items.append({"id": f"{cat}_{i:04d}", "group": cat, "instruction": f"do {cat}/{i}"})
    return items


def test_make_subset_produces_exact_per_category_count():
    items = _mock_items()
    subset = make_subset(items, n=200, stratify_key="group", seed=42)
    assert len(subset) == 200
    counts: dict[str, int] = {}
    for it in subset:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    assert counts == {"Dev": 40, "Creative": 40, "CAD": 40, "Scientific": 40, "Office": 40}


def test_make_subset_is_deterministic_across_calls():
    items = _mock_items()
    a = make_subset(items, n=200, stratify_key="group", seed=42)
    b = make_subset(items, n=200, stratify_key="group", seed=42)
    assert [it["id"] for it in a] == [it["id"] for it in b]


def test_make_subset_seed_changes_selection():
    items = _mock_items()
    a = make_subset(items, n=200, stratify_key="group", seed=42)
    b = make_subset(items, n=200, stratify_key="group", seed=43)
    a_ids = sorted(it["id"] for it in a)
    b_ids = sorted(it["id"] for it in b)
    assert a_ids != b_ids, "different seeds should pick different items"


def test_make_subset_picks_only_from_input():
    items = _mock_items()
    valid_ids = {it["id"] for it in items}
    subset = make_subset(items, n=200, stratify_key="group", seed=42)
    for it in subset:
        assert it["id"] in valid_ids


def test_subset_persistence_round_trip(tmp_path: Path):
    items = _mock_items()
    subset_path = tmp_path / "subset_200.jsonl"
    subset = make_subset(items, n=200, stratify_key="group", seed=42, persist_to=subset_path)
    assert subset_path.exists()

    # Reload and verify identical ids in the same order.
    reloaded = load_subset(subset_path)
    assert [it["id"] for it in reloaded] == [it["id"] for it in subset]


def test_subset_load_returns_same_items_as_make(tmp_path: Path):
    """Calling make_subset a second time with persist_to set should reuse the
    persisted file and return identical ids — across multiple invocations."""
    items = _mock_items()
    subset_path = tmp_path / "subset_200.jsonl"
    first = make_subset(items, n=200, stratify_key="group", seed=42, persist_to=subset_path)
    second = make_subset(items, n=200, stratify_key="group", seed=42, persist_to=subset_path)
    assert [it["id"] for it in first] == [it["id"] for it in second]


def test_make_subset_smaller_than_n_per_category_raises():
    """Asking for 200 across 5 cats but feeding only 30 in one cat should error."""
    items = []
    for cat in ("Dev", "Creative", "CAD", "Scientific", "Office"):
        n = 30 if cat == "Dev" else 300
        for i in range(n):
            items.append({"id": f"{cat}_{i:04d}", "group": cat})
    with pytest.raises(ValueError):
        make_subset(items, n=200, stratify_key="group", seed=42)


def test_make_subset_handles_uneven_division():
    """n=200 across 6 categories: should distribute as evenly as possible (33 or 34 per cat)."""
    items = []
    for cat in ("Dev", "Creative", "CAD", "Scientific", "Office", "OS"):
        for i in range(300):
            items.append({"id": f"{cat}_{i:04d}", "group": cat})
    subset = make_subset(items, n=200, stratify_key="group", seed=42)
    assert len(subset) == 200
    counts: dict[str, int] = {}
    for it in subset:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    # 200/6 = 33.33 -> distribution like [34,34,33,33,33,33]
    assert sum(counts.values()) == 200
    assert all(c in (33, 34) for c in counts.values())
