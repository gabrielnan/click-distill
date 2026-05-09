"""Stratified subset sampler for ScreenSpot-Pro intermediate eval.

Why this exists
---------------
The hackathon trains multiple LoRA adapters in 2hr (one per
streaming-snapshot tick). Full ScreenSpot-Pro eval = 1581 items, ~10 min on
1xH100. That doesn't fit the budget.

Instead: pick a stratified 200-item slice ONCE, persist it to
``eval/data/screenspot_pro_subset_200.jsonl``, and use it for every
intermediate eval. Run the full 1581-item eval only on the best-looking
adapter at the end (see ``eval/finalize.py``).

Stratification rule
-------------------
We sample ``ceil(n / k)`` items from each of ``k`` categories (down to
``floor(n / k)`` for the trailing categories), so every category is
represented roughly equally — this avoids the failure mode where the subset
acc looks great but the full eval regresses because we missed an entire
category. Item selection within each category is a deterministic
``random.Random(seed)`` shuffle, so the subset is bit-stable across runs and
machines given the same seed.

Persistence semantics
---------------------
* If ``persist_to`` is None, the subset is computed and returned in memory only.
* If ``persist_to`` is a path AND the file exists, we reload it and return
  THAT — guarantees that all training runs see identical items even if the
  source ``items`` order changes (e.g. annotation files are re-merged).
* If ``persist_to`` is a path and the file does not exist, we compute, write,
  and return.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _allocate_per_category(n: int, k: int) -> list[int]:
    """Distribute ``n`` items across ``k`` buckets as evenly as possible.

    For n=200, k=5 -> [40, 40, 40, 40, 40].
    For n=200, k=6 -> [34, 34, 33, 33, 33, 33] (leading buckets get the +1).
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    base = n // k
    extra = n % k
    return [base + (1 if i < extra else 0) for i in range(k)]


def _group_by(items: Iterable[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        grouped[it[key]].append(it)
    return dict(grouped)


def make_subset(
    items: list[dict],
    n: int,
    stratify_key: str,
    seed: int,
    persist_to: Path | None = None,
) -> list[dict]:
    """Return a deterministic stratified subset of ``items``.

    Args:
        items: full pool to sample from.
        n: target subset size.
        stratify_key: key in each item to stratify on (e.g. ``"group"``).
        seed: RNG seed; same seed + same items => same subset.
        persist_to: if provided, write the subset to this path (JSONL) and on
            subsequent calls with the same path, RELOAD the persisted file
            instead of recomputing — this is what guarantees cross-run
            stability of the eval slice.

    Returns:
        A new list of length ``n``. Items are not deep-copied (cheap reference).

    Raises:
        ValueError: if any category has fewer items than its allocated quota,
            or if the inputs are otherwise malformed.
    """
    # Reuse persisted subset if present — this is the cross-run stability
    # guarantee. We trust the file's contents over recomputation.
    if persist_to is not None and persist_to.exists():
        return load_subset(persist_to)

    grouped = _group_by(items, stratify_key)
    cats = sorted(grouped.keys())  # deterministic category order
    if not cats:
        raise ValueError("no items to subset (empty input)")

    quotas = _allocate_per_category(n, len(cats))

    rng = random.Random(seed)
    picked: list[dict] = []
    for cat, quota in zip(cats, quotas):
        pool = grouped[cat]
        if len(pool) < quota:
            raise ValueError(
                f"category {cat!r} has only {len(pool)} items, need {quota} "
                f"(reduce n or rebalance source data)"
            )
        # Sort by id (or full repr fallback) to make the shuffle deterministic
        # regardless of source iteration order.
        pool_sorted = sorted(pool, key=lambda it: it.get("id") or json.dumps(it, sort_keys=True))
        rng.shuffle(pool_sorted)
        picked.extend(pool_sorted[:quota])

    if persist_to is not None:
        persist_to.parent.mkdir(parents=True, exist_ok=True)
        with persist_to.open("w", encoding="utf-8") as fh:
            for it in picked:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    return picked


def load_subset(path: Path) -> list[dict]:
    """Load a previously-persisted subset JSONL."""
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build a stratified eval subset.")
    p.add_argument("--input", type=Path, required=True,
                   help="Input JSONL (e.g. eval/data/screenspot_pro/screenspot_pro.jsonl)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL of the stratified subset")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--stratify-key", default="group")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items = []
    with args.input.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    subset = make_subset(
        items,
        n=args.n,
        stratify_key=args.stratify_key,
        seed=args.seed,
        persist_to=args.output,
    )
    counts: dict[str, int] = defaultdict(int)
    for it in subset:
        counts[it[args.stratify_key]] += 1
    print(f"Wrote {len(subset)} items -> {args.output}")
    for cat, c in sorted(counts.items()):
        print(f"  {cat:12s} {c:4d}")


if __name__ == "__main__":
    _main()
