"""Full-eval finalizer for the streaming RSSD pipeline.

Workflow at the end of the hackathon (or during the last 15 min):

  1. Read ``results/leaderboard.json``.
  2. Pick the top-ranked adapter by subset eval accuracy (skip baseline).
  3. Run the FULL 1581-item ScreenSpot-Pro eval on that adapter.
  4. Write ``results/final.json`` with both subset acc (leaderboard ground
     truth) and full-eval acc — so the human can sanity-check that the
     subset was predictive.

The full-eval invocation is INJECTED via ``eval_fn`` so the test suite can
mock it. The CLI (``python eval/finalize.py``) wires it to a subprocess
call into ``eval/eval_screenspot_pro.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# Repo-root path so we can import siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.leaderboard import load_results  # noqa: E402


EvalFn = Callable[[str, str, Path], dict]


def pick_best_adapter(results_dir: Path) -> Optional[str]:
    """Pick the leaderboard's top adapter by overall acc, skipping baseline.

    Returns the run_name (e.g. ``"lora_1200"``) or None if no adapter rows
    are present.
    """
    rows = load_results(results_dir)
    candidates = [r for r in rows if r["run_name"] != "baseline"]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["overall_acc"], reverse=True)
    return candidates[0]["run_name"]


def _subset_acc_for(results_dir: Path, run_name: str) -> Optional[float]:
    for r in load_results(results_dir):
        if r["run_name"] == run_name:
            return r["overall_acc"]
    return None


def finalize_run(
    results_dir: Path,
    adapters_dir: Path,
    base_model: str,
    eval_fn: EvalFn,
) -> dict:
    """Run the full eval on the top-ranked adapter and write ``final.json``.

    Args:
        results_dir: dir containing ``leaderboard.json`` (and where ``final.json``
            will be written).
        adapters_dir: dir containing ``lora_<id>/`` subdirs.
        base_model: HF id or local path of the base model.
        eval_fn: callable ``(model, lora_path, output_path) -> dict``. Returns
            the full-eval payload (matching ``eval_screenspot_pro.build_payload``).

    Returns:
        The dict written to ``final.json``.

    Raises:
        RuntimeError: if no adapter rows are present in the leaderboard.
    """
    results_dir = Path(results_dir)
    adapters_dir = Path(adapters_dir)

    best = pick_best_adapter(results_dir)
    if best is None:
        raise RuntimeError(
            f"no adapters in leaderboard at {results_dir}/leaderboard.json"
        )

    lora_path = adapters_dir / best
    output = results_dir / f"final_eval_{best}.json"

    print(f"[finalize] running full eval on {best} (lora={lora_path})", flush=True)
    full = eval_fn(base_model, str(lora_path), output)

    final = {
        "adapter": best,
        "lora_path": str(lora_path),
        "subset_overall_acc": _subset_acc_for(results_dir, best),
        "full_eval": full,
    }
    final_path = results_dir / "final.json"
    final_path.write_text(json.dumps(final, indent=2))

    overall = (full.get("metrics") or {}).get("overall") or {}
    print(
        f"[finalize] {best}: subset={final['subset_overall_acc']:.3f}  "
        f"full={overall.get('acc', 0.0):.3f}  ({overall.get('n', 0)} items)",
        flush=True,
    )
    print(f"[finalize] wrote {final_path}", flush=True)
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _subprocess_eval_fn(eval_script: Path) -> EvalFn:
    """Default eval_fn: shells out to ``eval/eval_screenspot_pro.py``."""

    def _run(model: str, lora_path: str, output: Path) -> dict:
        cmd = [
            sys.executable, str(eval_script),
            "--model", model,
            "--lora", lora_path,
            "--output", str(output),
        ]
        print(f"[finalize] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        return json.loads(output.read_text())

    return _run


def _main() -> None:
    p = argparse.ArgumentParser(description="Full-eval finalizer.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--adapters-dir", type=Path, default=Path("adapters"))
    p.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    p.add_argument("--eval-script", type=Path,
                   default=Path(__file__).resolve().parent / "eval_screenspot_pro.py")
    args = p.parse_args()

    finalize_run(
        results_dir=args.results_dir,
        adapters_dir=args.adapters_dir,
        base_model=args.base_model,
        eval_fn=_subprocess_eval_fn(args.eval_script),
    )


if __name__ == "__main__":
    _main()
