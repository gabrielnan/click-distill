"""Leaderboard for the streaming RSSD pipeline.

After every training run the orchestrator (``train_run.sh``) runs the
subset eval and ``append_result()``s a row here. We keep two artefacts in
``results/``:

* ``leaderboard.json`` — the source-of-truth list of result dicts. Updated
  in place, deduped on ``run_name`` (latest write wins).
* ``leaderboard.txt`` — fixed-width pretty render. Re-derived from the JSON
  on every append, so it always reflects the current state of the JSON db.

Sort rule:
  1. ``baseline`` always at the top (so the human can spot deltas at a glance).
  2. Everything else by ``overall_acc`` descending.

Deltas:
  * Render relative to baseline if it exists.
  * Render an em-dash when there is no baseline, or for the baseline row itself.
"""

from __future__ import annotations

import json
from pathlib import Path

LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_TXT = "leaderboard.txt"

EM_DASH = "—"


def load_results(results_dir: Path) -> list[dict]:
    """Load the leaderboard JSON db (or [] if it doesn't exist)."""
    path = Path(results_dir) / LEADERBOARD_JSON
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _save_results(results_dir: Path, rows: list[dict]) -> None:
    path = Path(results_dir) / LEADERBOARD_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def _format_trained_on(v) -> str:
    if v is None:
        return EM_DASH
    return str(v)


def _format_delta(overall: float, baseline: float | None, is_baseline: bool) -> str:
    if is_baseline or baseline is None:
        return EM_DASH
    d = overall - baseline
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d):.3f}"


def format_leaderboard(rows: list[dict]) -> str:
    """Render a list of result dicts as a fixed-width text table.

    Required keys per row: run_name, overall_acc, text_acc, icon_acc,
    subset_size, trained_on.

    Sort: baseline first if present, then others by overall_acc DESC.
    """
    if not rows:
        return ""

    # Find baseline (if any) for delta computation.
    baseline_row = next((r for r in rows if r["run_name"] == "baseline"), None)
    baseline_overall = baseline_row["overall_acc"] if baseline_row else None

    # Sort.
    others = [r for r in rows if r["run_name"] != "baseline"]
    others.sort(key=lambda r: r["overall_acc"], reverse=True)
    ordered = ([baseline_row] if baseline_row else []) + others

    # Column widths.
    cols = ("run", "overall", "delta", "text", "icon", "subset_size", "trained_on")
    widths = {
        "run": max(14, max(len(r["run_name"]) for r in ordered) + 2),
        "overall": 8,
        "delta": 7,
        "text": 6,
        "icon": 6,
        "subset_size": 12,
        "trained_on": 12,
    }

    def fmt_row(name, overall, delta, text, icon, subset_size, trained_on):
        return (
            f"{name:<{widths['run']}}"
            f"{overall:<{widths['overall']}}"
            f"{delta:<{widths['delta']}}"
            f"{text:<{widths['text']}}"
            f"{icon:<{widths['icon']}}"
            f"{subset_size:<{widths['subset_size']}}"
            f"{trained_on:<{widths['trained_on']}}"
        )

    lines = [fmt_row(*cols)]
    for r in ordered:
        is_baseline = r["run_name"] == "baseline"
        lines.append(fmt_row(
            r["run_name"],
            f"{r['overall_acc']:.3f}",
            _format_delta(r["overall_acc"], baseline_overall, is_baseline),
            f"{r['text_acc']:.3f}",
            f"{r['icon_acc']:.3f}",
            str(r.get("subset_size", "")),
            _format_trained_on(r.get("trained_on")),
        ))
    return "\n".join(lines) + "\n"


def append_result(results_dir: Path, run_name: str, eval_result: dict) -> None:
    """Append/replace a row in the leaderboard.

    ``eval_result`` is a dict with at least: ``overall_acc``, ``text_acc``,
    ``icon_acc``, ``subset_size``. It may include ``trained_on`` (number of
    SFT items the adapter trained on) and ``run_name`` (overrides the
    positional arg if given).

    Re-running the same ``run_name`` overwrites the previous entry.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(results_dir)

    # Build the new row, defaulting run_name from the arg.
    new_row = dict(eval_result)
    new_row["run_name"] = new_row.get("run_name") or run_name

    # Replace any existing row with the same name.
    rows = [r for r in rows if r["run_name"] != new_row["run_name"]]
    rows.append(new_row)

    _save_results(results_dir, rows)

    text = format_leaderboard(rows)
    (results_dir / LEADERBOARD_TXT).write_text(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Print the leaderboard.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        print(f"(no results yet in {args.results_dir})")
        return
    print(format_leaderboard(rows))


if __name__ == "__main__":
    _main()
