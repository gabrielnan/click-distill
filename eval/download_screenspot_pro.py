"""Download the ScreenSpot-Pro test set to eval/data/screenspot_pro/.

Dataset: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
Layout (mirrors the upstream HF repo):

    eval/data/screenspot_pro/
      annotations/        # 27 per-app .json files (each is a list of items)
      images/             # subfolders per application (e.g. vscode_mac/...)
      screenspot_pro.jsonl  # flattened single-file index for fast iteration

Each line of `screenspot_pro.jsonl` is one item:

    {
      "id": "vscode_macos_0",
      "image": "images/vscode_mac/screenshot_2024-12-03_15-15-02.png",  # repo-relative
      "instruction": "Refresh the file explorer.",
      "bbox": [473, 183, 503, 219],   # xyxy in source-image pixels
      "img_size": [2560, 1664],        # [width, height]
      "application": "vscode",
      "platform": "macos",
      "group": "Dev",
      "ui_type": "icon",
      "task_filename": "vscode_macos"
    }

Idempotent: re-runs skip the snapshot download if the cache already has the
files, and rebuild the JSONL in place. ~3.4 GB total on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DATASET_REPO = "likaixin/ScreenSpot-Pro"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "screenspot_pro"


def download(out_dir: Path) -> Path:
    """Snapshot-download the dataset repo into `out_dir`. Returns the local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        sys.exit(
            "huggingface_hub not installed. Run: pip install huggingface_hub\n"
            f"(import error: {e})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_REPO} -> {out_dir} (idempotent, ~3.4 GB)")
    local_path = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(out_dir),
        # Only fetch what we need; skip annotate.html etc. is fine but cheap.
        allow_patterns=["annotations/*.json", "images/**", "README.md", "eval.yaml"],
    )
    return Path(local_path)


def build_jsonl(root: Path, jsonl_path: Path) -> int:
    """Concatenate every annotation file into a single JSONL. Returns item count."""
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        sys.exit(f"Missing annotations dir: {ann_dir}")

    n = 0
    missing_imgs: list[str] = []
    with jsonl_path.open("w") as out:
        for ann_file in sorted(ann_dir.glob("*.json")):
            task_filename = ann_file.stem  # e.g. "vscode_macos"
            with ann_file.open() as f:
                items = json.load(f)
            for item in items:
                img_rel = Path("images") / item["img_filename"]
                if not (root / img_rel).is_file():
                    missing_imgs.append(str(img_rel))
                record = {
                    "id": item["id"],
                    "image": str(img_rel),
                    "instruction": item["instruction"],
                    "instruction_cn": item.get("instruction_cn"),
                    "bbox": item["bbox"],          # xyxy pixels
                    "img_size": item["img_size"],  # [w, h]
                    "application": item.get("application"),
                    "platform": item.get("platform"),
                    "group": item.get("group"),
                    "ui_type": item.get("ui_type"),
                    "task_filename": task_filename,
                }
                out.write(json.dumps(record) + "\n")
                n += 1

    if missing_imgs:
        # Don't fail hard; many CUA datasets ship a handful of broken refs.
        print(
            f"WARNING: {len(missing_imgs)} referenced images missing on disk "
            f"(first 3: {missing_imgs[:3]})",
            file=sys.stderr,
        )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the snapshot download; only rebuild the JSONL "
                             "(useful if you already have the files)")
    args = parser.parse_args()

    out_dir: Path = args.out
    if not args.skip_download:
        download(out_dir)
    else:
        if not out_dir.is_dir():
            sys.exit(f"--skip-download but {out_dir} doesn't exist")

    jsonl = out_dir / "screenspot_pro.jsonl"
    n = build_jsonl(out_dir, jsonl)
    print(f"Wrote {n} items -> {jsonl}")
    print(f"Done. To eval: python eval/eval_screenspot_pro.py "
          f"--model Tzafon/Northstar-CUA-Fast --data {out_dir}")


if __name__ == "__main__":
    main()
