"""Filter Northstar samples by ground-truth bbox to build SFT data.

Input: raw_samples.jsonl (output of sample_northstar.py).
Output: sft_data.jsonl, one line per kept (image, instruction, completion).

Parsing strategy (in order of preference, falls back through them):
  1. ``<tool_call>{"name":"computer_use","arguments":{...}}</tool_call>``
     — the canonical Qwen3-VL tool-call format the chat template trains the
     model to emit. We pull ``arguments.x``, ``arguments.y`` from the JSON.
  2. Bare JSON object ``{"name":..., "arguments":{"x":..,"y":..}}`` — model
     sometimes drops the XML tags.
  3. Bare ``{"type":"click","x":..,"y":..}`` arguments-only JSON.
  4. Plain ``click(x, y)`` style call (covers any pretrain residue).

Coordinate convention: model emits 0-999 normalized. We denormalize to pixels
using each item's image_width / image_height (matches lightcone's
``COORD_KEYS`` mapping in ``examples/_cua.py``: ``x = x/1000 * display_width``).

Bbox is xyxy in pixel space (per the data generator contract in PLAN.md).

Usage:
    python sample/filter_by_bbox.py \\
        --input sample/raw_samples.jsonl \\
        --output sample/sft_data.jsonl \\
        --keep first
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# Use the canonical parser + denormalize from the repo-root module so the
# eval and sample pipelines agree byte-for-byte on extraction.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from northstar_format import parse_action, denormalize  # noqa: E402

# --- Action parsing (thin adapter over canonical parse_action) --------------

# Tracks which parser path fired — kept as a single bucket since the canonical
# parser handles all formats internally and doesn't expose per-path provenance.
ParseResult = tuple[float, float, str, str]  # (x, y, action_type, parser_name)


def parse_click(completion: str) -> ParseResult | None:
    """Extract the first click-ish action from a completion string.

    Returns (x, y, action_type, parser_name) or None.
    Coordinates returned in the model's emitted space (0-999 normalized).
    Filters down to point-style click actions only (drag etc. are dropped).
    """
    parsed = parse_action(completion)
    if parsed is None:
        return None
    if parsed["type"] not in {"click", "double_click", "triple_click", "right_click"}:
        return None
    return float(parsed["x"]), float(parsed["y"]), parsed["type"], "northstar_format"


# --- Bbox check -------------------------------------------------------------

def denorm_to_pixels(x_norm: float, y_norm: float, img_w: int, img_h: int) -> tuple[float, float]:
    """0-999 normalized -> pixel coords. Delegates to canonical ``denormalize``
    (uses ``int(coord / 1000 * dim)`` exactly like lightcone)."""
    px, py = denormalize(int(round(x_norm)), int(round(y_norm)), img_w, img_h)
    return float(px), float(py)


def in_bbox(px: float, py: float, bbox_xyxy: list[float]) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= px <= x2 and y1 <= py <= y2


# --- Main -------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep", choices=["first", "all"], default="first",
        help="'first' (default) keeps the first hit per item — avoids one item dominating SFT. "
             "'all' keeps every hit (for diversity-aware training).",
    )
    parser.add_argument("--verbose-misses", action="store_true",
                        help="Print each parse failure / bbox miss (noisy; debug only).")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_items = 0
    n_items_with_hit = 0
    total_completions = 0
    total_parse_ok = 0
    total_parse_fail = 0
    total_bbox_hit = 0
    total_bbox_miss = 0
    parser_counts: Counter[str] = Counter()
    hits_per_item: list[int] = []

    out_fh = args.output.open("w", encoding="utf-8")
    try:
        for item in iter_jsonl(args.input):
            n_items += 1
            samples: list[str] = item.get("samples", [])
            bbox = item["bbox"]
            img_w = int(item["image_width"])
            img_h = int(item["image_height"])

            item_hits = 0
            for completion in samples:
                total_completions += 1
                parsed = parse_click(completion)
                if parsed is None:
                    total_parse_fail += 1
                    if args.verbose_misses:
                        print(f"[parse-fail] {item.get('item_id')}: {completion[:120]!r}")
                    continue
                x_norm, y_norm, _action_type, parser_name = parsed
                total_parse_ok += 1
                parser_counts[parser_name] += 1

                px, py = denorm_to_pixels(x_norm, y_norm, img_w, img_h)
                if in_bbox(px, py, bbox):
                    total_bbox_hit += 1
                    item_hits += 1
                    record = {
                        "image_path": item["image_path"],
                        "instruction": item["instruction"],
                        "completion": completion,
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if args.keep == "first":
                        # Skip remaining samples for this item.
                        break
                else:
                    total_bbox_miss += 1
                    if args.verbose_misses:
                        print(
                            f"[bbox-miss] {item.get('item_id')}: "
                            f"norm=({x_norm:.0f},{y_norm:.0f}) "
                            f"pixel=({px:.1f},{py:.1f}) bbox={bbox}"
                        )

            if item_hits > 0:
                n_items_with_hit += 1
            hits_per_item.append(item_hits)
    finally:
        out_fh.close()

    # --- Stats ---
    yield_pct = 100.0 * n_items_with_hit / max(n_items, 1)
    parse_fail_pct = 100.0 * total_parse_fail / max(total_completions, 1)
    bbox_miss_pct = 100.0 * total_bbox_miss / max(total_parse_ok, 1)
    mean_hits = sum(hits_per_item) / max(n_items, 1)

    print("=" * 60)
    print(f"Filter stats (input: {args.input})")
    print("=" * 60)
    print(f"Items                     : {n_items}")
    print(f"  with >=1 hit            : {n_items_with_hit}  ({yield_pct:.1f}% yield)")
    print(f"  mean hits / item        : {mean_hits:.2f}")
    print(f"Completions               : {total_completions}")
    print(f"  parsed OK               : {total_parse_ok}  ({100 - parse_fail_pct:.1f}%)")
    print(f"  parse failures          : {total_parse_fail}  ({parse_fail_pct:.1f}%)")
    print(f"  bbox hits               : {total_bbox_hit}")
    print(f"  bbox misses (parsed)    : {total_bbox_miss}  ({bbox_miss_pct:.1f}% of parsed)")
    print(f"Parser breakdown          : {dict(parser_counts)}")
    print(f"Wrote {total_bbox_hit if args.keep == 'all' else n_items_with_hit} "
          f"records to {args.output} (--keep {args.keep})")


# --- Sanity test ------------------------------------------------------------

def _self_test() -> None:
    """Synthetic completions covering each parser path. Run via:
        python -c "from sample.filter_by_bbox import _self_test; _self_test()"

    Note: the canonical parser lives in northstar_format.parse_action; this
    test asserts the (x, y, type) tuple it returns via the parse_click adapter.
    """
    cases = [
        # (completion, expected (x, y, action_type) or None)
        (
            'I will click the search bar.\n<tool_call>\n{"name": "computer_use", '
            '"arguments": {"type": "click", "x": 500, "y": 250}}\n</tool_call>',
            (500.0, 250.0, "click"),
        ),
        (
            '<tool_call>{"name":"computer_use","arguments":{"type":"double_click","x":100,"y":200}}</tool_call>',
            (100.0, 200.0, "double_click"),
        ),
        (
            '{"name":"computer_use","arguments":{"type":"click","x":42,"y":99}}',
            (42.0, 99.0, "click"),
        ),
        (
            '{"type": "click", "x": 12, "y": 34}',
            (12.0, 34.0, "click"),
        ),
        (
            '{"type": "click", "y": 34, "x": 12}',
            (12.0, 34.0, "click"),
        ),
        (
            'I cannot see a button to click.',
            None,
        ),
        # Tool call with stringified arguments (some templates do this).
        (
            '<tool_call>{"name":"computer_use","arguments":"{\\"type\\":\\"click\\",\\"x\\":7,\\"y\\":8}"}</tool_call>',
            (7.0, 8.0, "click"),
        ),
    ]
    failures = 0
    for completion, expected in cases:
        got = parse_click(completion)
        if expected is None:
            ok = got is None
            simplified = None
        else:
            ok = (
                got is not None
                and got[0] == expected[0]
                and got[1] == expected[1]
                and got[2] == expected[2]
            )
            simplified = (got[0], got[1], got[2]) if got else None
        status = "OK" if ok else "FAIL"
        print(f"[{status}] expected={expected} got={simplified}  src={completion[:60]!r}")
        if not ok:
            failures += 1
    if failures:
        sys.exit(f"{failures} self-test(s) failed")
    print("All self-tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
