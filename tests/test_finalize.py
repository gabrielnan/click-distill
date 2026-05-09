"""Full-eval finalizer test (Component 7).

After the hackathon's training-run loop, ``finalize.py`` reads the
leaderboard, picks the top-ranked adapter (by subset-eval acc), runs the
FULL 1581-item eval on it (mocked here), and writes ``results/final.json``
plus a printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.finalize import finalize_run, pick_best_adapter
from eval.leaderboard import append_result


def _seed_leaderboard(results_dir: Path):
    append_result(results_dir, "baseline",
                  {"overall_acc": 0.310, "text_acc": 0.412, "icon_acc": 0.218,
                   "subset_size": 200, "trained_on": None})
    append_result(results_dir, "lora_1130",
                  {"overall_acc": 0.358, "text_acc": 0.461, "icon_acc": 0.262,
                   "subset_size": 200, "trained_on": 6024})
    append_result(results_dir, "lora_1200",
                  {"overall_acc": 0.371, "text_acc": 0.475, "icon_acc": 0.273,
                   "subset_size": 200, "trained_on": 11423})
    append_result(results_dir, "lora_1230",
                  {"overall_acc": 0.342, "text_acc": 0.450, "icon_acc": 0.250,
                   "subset_size": 200, "trained_on": 15000})


def test_pick_best_adapter_skips_baseline(tmp_path: Path):
    _seed_leaderboard(tmp_path)
    best = pick_best_adapter(tmp_path)
    assert best == "lora_1200"


def test_pick_best_adapter_only_baseline_returns_none(tmp_path: Path):
    append_result(tmp_path, "baseline",
                  {"overall_acc": 0.310, "text_acc": 0.412, "icon_acc": 0.218,
                   "subset_size": 200, "trained_on": None})
    assert pick_best_adapter(tmp_path) is None


def test_finalize_run_invokes_full_eval_on_top_adapter(tmp_path: Path):
    """We mock the eval runner: assert it gets called with the right adapter
    and writes the expected final.json."""
    _seed_leaderboard(tmp_path)
    adapters_dir = tmp_path / "adapters"
    (adapters_dir / "lora_1200").mkdir(parents=True)

    captured = {}

    def fake_eval(model: str, lora_path: str, output: Path) -> dict:
        captured["model"] = model
        captured["lora_path"] = lora_path
        captured["output"] = output
        return {
            "model": model,
            "lora": lora_path,
            "n_items": 1581,
            "subset_size": None,  # full eval
            "metrics": {"overall": {"acc": 0.392, "n": 1581, "wrong_format": 5}},
        }

    final = finalize_run(
        results_dir=tmp_path,
        adapters_dir=adapters_dir,
        base_model="Tzafon/Northstar-CUA-Fast",
        eval_fn=fake_eval,
    )

    assert captured["lora_path"] == str(adapters_dir / "lora_1200")
    assert captured["model"] == "Tzafon/Northstar-CUA-Fast"
    final_path = tmp_path / "final.json"
    assert final_path.is_file()
    payload = json.loads(final_path.read_text())
    assert payload["adapter"] == "lora_1200"
    assert payload["full_eval"]["n_items"] == 1581
    assert payload["full_eval"]["metrics"]["overall"]["acc"] == 0.392
    # Sanity: returned dict matches written file.
    assert final["adapter"] == "lora_1200"


def test_finalize_run_no_adapters_raises(tmp_path: Path):
    append_result(tmp_path, "baseline",
                  {"overall_acc": 0.310, "text_acc": 0.412, "icon_acc": 0.218,
                   "subset_size": 200, "trained_on": None})
    with pytest.raises(RuntimeError):
        finalize_run(tmp_path, tmp_path / "adapters",
                     base_model="x", eval_fn=lambda *a, **k: {})


def test_finalize_run_records_subset_acc_alongside_full(tmp_path: Path):
    """The final.json should keep both the subset acc (from the leaderboard)
    AND the full-eval acc, so we can sanity-check the subset's predictive power."""
    _seed_leaderboard(tmp_path)
    adapters_dir = tmp_path / "adapters"
    (adapters_dir / "lora_1200").mkdir(parents=True)

    def fake_eval(model, lora_path, output):
        return {"model": model, "lora": lora_path, "n_items": 1581,
                "subset_size": None,
                "metrics": {"overall": {"acc": 0.392, "n": 1581, "wrong_format": 5}}}

    final = finalize_run(
        tmp_path, adapters_dir,
        base_model="Tzafon/Northstar-CUA-Fast", eval_fn=fake_eval,
    )
    # Subset acc was 0.371 for lora_1200 — must be in the payload.
    assert final["subset_overall_acc"] == 0.371
    # Full-eval overall is 0.392.
    assert final["full_eval"]["metrics"]["overall"]["acc"] == 0.392
