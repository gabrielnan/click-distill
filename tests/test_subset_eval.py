"""Subset-aware eval test (Component 4).

The intermediate eval should load a 200-item stratified slice instead of the
full 1581-item dataset. We expose this via a ``--subset N`` flag plus a
helper function ``select_subset_items()`` that the orchestrator can call
without needing vLLM.

We don't need a GPU for this: the heavy backend is constructed only when CLI
``main()`` runs. The selection logic is a pure function over a list of dicts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add eval/ dir to sys.path so we can import the script as a module.
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_screenspot_pro import select_subset_items  # noqa: E402


def _items(per_cat: int = 300) -> list[dict]:
    items = []
    for cat in ("Dev", "Creative", "CAD", "Scientific", "Office"):
        for i in range(per_cat):
            items.append({
                "id": f"{cat}_{i:04d}",
                "image": f"images/{cat}/img{i}.png",
                "instruction": f"do {cat}/{i}",
                "bbox": [0, 0, 10, 10],
                "img_size": [1024, 768],
                "group": cat,
                "ui_type": "icon" if i % 2 == 0 else "text",
            })
    return items


def test_select_subset_items_uses_persisted_file_if_present(tmp_path: Path):
    """Once the subset has been persisted, subsequent calls reload it
    verbatim — no recomputation, no chance of drift."""
    items = _items()
    subset_path = tmp_path / "subset_200.jsonl"

    # First call: writes the file.
    out1 = select_subset_items(items, n=200, subset_path=subset_path)
    assert len(out1) == 200
    assert subset_path.is_file()

    # Mutate the in-memory items list — the second call must IGNORE it
    # (because it reloads from disk).
    items[0] = {**items[0], "instruction": "MUTATED"}
    out2 = select_subset_items(items, n=200, subset_path=subset_path)
    # If reloaded from disk, none should carry the MUTATED instruction.
    assert all(it["instruction"] != "MUTATED" for it in out2)


def test_select_subset_items_stratified_balance(tmp_path: Path):
    items = _items()
    subset_path = tmp_path / "subset_200.jsonl"
    out = select_subset_items(items, n=200, subset_path=subset_path)
    assert len(out) == 200
    counts: dict[str, int] = {}
    for it in out:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    assert counts == {"Dev": 40, "Creative": 40, "CAD": 40, "Scientific": 40, "Office": 40}


def test_select_subset_items_deterministic_across_calls(tmp_path: Path):
    items = _items()
    s1 = tmp_path / "a.jsonl"
    s2 = tmp_path / "b.jsonl"
    a = select_subset_items(items, n=200, subset_path=s1)
    b = select_subset_items(items, n=200, subset_path=s2)
    # Same inputs + default seed -> same selection (by id).
    assert [it["id"] for it in a] == [it["id"] for it in b]


def test_evaluate_includes_subset_size_in_payload(tmp_path: Path):
    """The output JSON of the eval CLI must record subset_size so the
    leaderboard can show it. We exercise the helper that builds the
    payload, NOT the full eval (which needs a GPU)."""
    from eval_screenspot_pro import build_payload  # noqa: WPS433

    fake_results = [
        {"id": "x", "correct": True, "wrong_format": False},
        {"id": "y", "correct": False, "wrong_format": False},
    ]
    payload = build_payload(
        model="Tzafon/Northstar-CUA-Fast",
        lora=None,
        results=fake_results,
        elapsed=1.23,
        subset_size=200,
    )
    assert payload["subset_size"] == 200
    assert payload["n_items"] == 2
    assert payload["model"] == "Tzafon/Northstar-CUA-Fast"
    assert "metrics" in payload


def test_build_payload_no_subset_size_field_omitted_or_null():
    """Full-eval calls don't pass subset_size — output should still be valid."""
    from eval_screenspot_pro import build_payload

    payload = build_payload(
        model="Tzafon/Northstar-CUA-Fast",
        lora=None,
        results=[],
        elapsed=0.0,
        subset_size=None,
    )
    # Either omitted or explicitly None — both are fine. Test the explicit-None case.
    assert payload.get("subset_size") is None
