"""Leaderboard test (Component 3).

The leaderboard tracks per-run subset-eval metrics so we can pick the best
adapter for the final 1581-item eval. We render a small fixed-width text
table to ``results/leaderboard.txt`` after each run; baseline first, the
rest sorted by overall acc descending, deltas relative to baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.leaderboard import append_result, format_leaderboard, load_results


def _result(name: str, overall: float, text: float, icon: float, *, n: int = 200,
            trained_on=None) -> dict:
    return {
        "run_name": name,
        "overall_acc": overall,
        "text_acc": text,
        "icon_acc": icon,
        "subset_size": n,
        "trained_on": trained_on,
    }


# ---------------------------------------------------------------------------
# format_leaderboard pure-function tests
# ---------------------------------------------------------------------------
def test_format_leaderboard_baseline_only():
    rows = [_result("baseline", 0.310, 0.412, 0.218)]
    out = format_leaderboard(rows)
    lines = out.strip().splitlines()
    # Header + 1 row.
    assert len(lines) == 2
    assert "run" in lines[0] and "overall" in lines[0]
    assert "baseline" in lines[1]
    assert "0.310" in lines[1]
    # Baseline has em-dash for delta and trained_on.
    assert "—" in lines[1]


def test_format_leaderboard_sorts_by_overall_desc_baseline_first():
    rows = [
        _result("baseline", 0.310, 0.412, 0.218),
        _result("lora_1130", 0.358, 0.461, 0.262, trained_on=6024),
        _result("lora_1200", 0.371, 0.475, 0.273, trained_on=11423),
        _result("lora_1230", 0.342, 0.450, 0.250, trained_on=15000),
    ]
    out = format_leaderboard(rows)
    body_lines = out.strip().splitlines()[1:]  # skip header
    names = [l.split()[0] for l in body_lines]
    # Baseline must always appear first.
    assert names[0] == "baseline"
    # The rest must be sorted by overall acc DESC (lora_1200 > lora_1130 > lora_1230)
    assert names[1:] == ["lora_1200", "lora_1130", "lora_1230"]


def test_format_leaderboard_deltas_relative_to_baseline():
    rows = [
        _result("baseline", 0.300, 0.400, 0.200),
        _result("lora_1130", 0.350, 0.450, 0.250, trained_on=6024),
    ]
    out = format_leaderboard(rows)
    lines = out.strip().splitlines()
    lora_line = next(l for l in lines if "lora_1130" in l)
    # Delta = +0.050 (allow rendering as +0.05 too — check substring).
    assert "+0.05" in lora_line


def test_format_leaderboard_no_baseline_omits_deltas():
    rows = [
        _result("lora_1130", 0.358, 0.461, 0.262, trained_on=6024),
        _result("lora_1200", 0.371, 0.475, 0.273, trained_on=11423),
    ]
    out = format_leaderboard(rows)
    lines = out.strip().splitlines()
    body = lines[1:]
    # Without baseline, delta column should render em-dash for every row.
    for line in body:
        assert "—" in line


def test_format_leaderboard_columns_present():
    rows = [_result("baseline", 0.310, 0.412, 0.218)]
    out = format_leaderboard(rows)
    header = out.strip().splitlines()[0]
    for col in ("run", "overall", "delta", "text", "icon", "subset_size", "trained_on"):
        assert col in header


# ---------------------------------------------------------------------------
# append_result side-effecting tests
# ---------------------------------------------------------------------------
def test_append_result_creates_file_and_json_db(tmp_path: Path):
    res = _result("baseline", 0.310, 0.412, 0.218)
    append_result(tmp_path, "baseline", res)

    txt = (tmp_path / "leaderboard.txt").read_text()
    assert "baseline" in txt and "0.310" in txt

    # Underlying JSON db (so re-formatting after appends doesn't lose history).
    db = (tmp_path / "leaderboard.json")
    assert db.is_file()
    rows = json.loads(db.read_text())
    assert len(rows) == 1
    assert rows[0]["run_name"] == "baseline"


def test_append_result_accumulates_runs(tmp_path: Path):
    append_result(tmp_path, "baseline", _result("baseline", 0.310, 0.412, 0.218))
    append_result(tmp_path, "lora_1130", _result("lora_1130", 0.358, 0.461, 0.262, trained_on=6024))
    append_result(tmp_path, "lora_1200", _result("lora_1200", 0.371, 0.475, 0.273, trained_on=11423))

    rows = load_results(tmp_path)
    names = [r["run_name"] for r in rows]
    assert sorted(names) == ["baseline", "lora_1130", "lora_1200"]

    txt = (tmp_path / "leaderboard.txt").read_text()
    body = [l for l in txt.splitlines() if l.strip()][1:]
    # Order: baseline first, then sorted by overall desc.
    assert body[0].split()[0] == "baseline"
    assert body[1].split()[0] == "lora_1200"
    assert body[2].split()[0] == "lora_1130"


def test_append_result_overwrites_same_run_name(tmp_path: Path):
    """If the same run name is appended twice, the second wins (e.g. user re-ran
    with a different subset). We don't want stale duplicates."""
    append_result(tmp_path, "lora_1130", _result("lora_1130", 0.300, 0.400, 0.200, trained_on=1000))
    append_result(tmp_path, "lora_1130", _result("lora_1130", 0.358, 0.461, 0.262, trained_on=6024))
    rows = load_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["overall_acc"] == 0.358
    assert rows[0]["trained_on"] == 6024
