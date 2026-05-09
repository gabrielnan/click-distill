"""Download + filter OS-Atlas grounding data into a 5k "hard subset" JSONL.

Source: https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data (Apache-2.0)

Strategy
--------
The full dataset is ~800GB (mostly images). We do NOT pull all images here.
We pull only the small per-domain annotation JSONs (each item = 1 screenshot,
N grounding elements with bbox + instruction). We expand to per-element rows,
filter to "hard" items (small bbox, instruction length 5-30 words,
non-mobile), sample 5k balanced across selected sources, and write JSONL.

The actual screenshot bytes must be fetched separately via the image archives
listed in the README and the runtime instructions printed at the end of this
script. The output JSONL stores the image's relative path under each domain's
extracted images directory; the downstream sampling script joins this with
`--images-root`.

Bbox is stored in NORMALIZED [0,1] xyxy coordinates. To convert to pixel
xyxy, multiply by image_width / image_height respectively. We do NOT fill
image_width / image_height here (would require downloading every image);
the downstream script fills them when it loads each PNG.

Usage
-----
    python data/download_osatlas.py                  # default 5k, all 3 desktop OSes
    python data/download_osatlas.py --n 10000
    python data/download_osatlas.py --sources desktop/linux,desktop/macos
    python data/download_osatlas.py --sources desktop/linux,desktop/macos,desktop/windows,web/seeclick
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

# ---------- constants ----------------------------------------------------------

HF_BASE = "https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data/resolve/main"

# (source_id, annotation_url_path, images_dir_after_extraction, image_archives)
# images_dir_after_extraction is what we prepend to img_filename in the JSONL.
# image_archives is the set of files that need to be downloaded + extracted on
# the GPU box to make the image referenced by img_filename resolvable.
SOURCES = {
    "desktop/linux": {
        "ann": "desktop_domain/linux_splited.json",
        "images_dir": "desktop_domain/linux_images",
        "archives": ["desktop_domain/linux_images.zip"],
    },
    "desktop/macos": {
        "ann": "desktop_domain/macos_splited.json",
        "images_dir": "desktop_domain/macos_images",
        "archives": ["desktop_domain/macos_images.zip"],
    },
    "desktop/windows": {
        "ann": "desktop_domain/windows_splited.json",
        "images_dir": "desktop_domain/windows_images",
        "archives": [
            "desktop_domain/windows_image_aa",
            "desktop_domain/windows_image_ab",
            "desktop_domain/windows_image_ac",
            "desktop_domain/windows_image_ad",
        ],
    },
    "web/seeclick": {
        "ann": "web_domain/seeclick_web.json",
        "images_dir": "web_domain/seeclick_web_images",
        "archives": [
            "web_domain/seeclick_web_image_aa",
            "web_domain/seeclick_web_image_ab",
            "web_domain/seeclick_web_image_ac",
            "web_domain/seeclick_web_image_ad",
        ],
    },
}

DEFAULT_SOURCES = "desktop/linux,desktop/macos,desktop/windows"

# Hard-subset filter knobs (per the plan).
MAX_BBOX_AREA_FRAC = 0.01  # < 1% of screen area
MIN_WORDS = 5
MAX_WORDS = 30


# ---------- helpers ------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def annotation_cache_path(cache_dir: Path, source_id: str) -> Path:
    return cache_dir / source_id.replace("/", "_") / "annotations.json"


def download_annotation(source_id: str, cache_dir: Path) -> Path:
    """Download the annotation JSON for a source if not already cached."""
    spec = SOURCES[source_id]
    out = annotation_cache_path(cache_dir, source_id)
    if out.exists() and out.stat().st_size > 0:
        logging.info("[%s] annotation already cached: %s (%.1f MB)",
                     source_id, out, out.stat().st_size / 1024 / 1024)
        return out

    url = f"{HF_BASE}/{spec['ann']}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    logging.info("[%s] downloading annotations from %s", source_id, url)
    t0 = time.time()
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("content-length", 0))
            read = 0
            chunk = 1024 * 1024
            last_log = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                read += len(buf)
                if total and read - last_log > 50 * 1024 * 1024:
                    pct = read / total * 100
                    logging.info("[%s] %.0f MB / %.0f MB (%.1f%%)",
                                 source_id, read / 1024 / 1024, total / 1024 / 1024, pct)
                    last_log = read
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"FATAL: HTTP {e.code} fetching {url}\n"
            f"  Verify the dataset still exists at "
            f"https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data "
            f"and that you have network access. Do NOT fall back to a "
            f"synthetic dataset."
        ) from e
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"FATAL: network error fetching {url}: {e}\n"
            f"  Check connectivity. Do NOT fall back to a synthetic dataset."
        ) from e
    tmp.rename(out)
    logging.info("[%s] downloaded %.1f MB in %.1fs",
                 source_id, out.stat().st_size / 1024 / 1024, time.time() - t0)
    return out


def iter_filtered_elements(
    source_id: str,
    ann_path: Path,
    skipped: Counter,
) -> Iterator[dict]:
    """Stream items from a single annotation JSON, expand to per-element rows,
    and apply the hard-subset filter. Yields dicts with the final JSONL schema."""
    spec = SOURCES[source_id]
    images_dir = spec["images_dir"]

    # Annotation JSONs are top-level lists. They're small enough to load whole
    # (largest is windows_splited.json at ~270MB), but we keep memory cheap by
    # not building a giant intermediate list.
    with open(ann_path, "rb") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise SystemExit(f"FATAL: expected a JSON list in {ann_path}, got {type(data).__name__}")

    for item in data:
        img = item.get("img_filename")
        elements = item.get("elements")
        if not img or not isinstance(elements, list):
            skipped["missing_image_or_elements"] += 1
            continue
        for el in elements:
            instr = el.get("instruction")
            bbox = el.get("bbox")
            if not instr or not isinstance(instr, str):
                skipped["missing_instruction"] += 1
                continue
            if not bbox or len(bbox) != 4:
                skipped["missing_or_bad_bbox"] += 1
                continue
            try:
                l, t, r, b = (float(v) for v in bbox)
            except (TypeError, ValueError):
                skipped["non_numeric_bbox"] += 1
                continue
            # bboxes should be normalized in [0,1]; reject out-of-range / inverted.
            if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
                skipped["invalid_bbox_geometry"] += 1
                continue
            area = (r - l) * (b - t)
            if area >= MAX_BBOX_AREA_FRAC:
                skipped["bbox_too_large"] += 1
                continue
            n_words = len(instr.split())
            if not (MIN_WORDS <= n_words <= MAX_WORDS):
                skipped["instruction_length"] += 1
                continue
            yield {
                "image_path": f"{images_dir}/{img}",
                "instruction": instr.strip(),
                "bbox": [l, t, r, b],          # normalized [0,1] xyxy
                "bbox_format": "normalized_xyxy",
                "image_width": None,           # filled by downstream loader
                "image_height": None,          # filled by downstream loader
                "source": source_id,
                "data_type": el.get("data_type"),
            }


def collect(
    sources: list[str],
    cache_dir: Path,
) -> tuple[list[dict], dict[str, Counter]]:
    """Download + filter all sources. Returns (rows, per-source skip counters)."""
    rows: list[dict] = []
    skip_by_source: dict[str, Counter] = {}
    for src in sources:
        if src not in SOURCES:
            raise SystemExit(
                f"FATAL: unknown source {src!r}. Known: {list(SOURCES)}"
            )
        ann_path = download_annotation(src, cache_dir)
        skipped: Counter = Counter()
        n_before = len(rows)
        last_log = n_before
        for row in iter_filtered_elements(src, ann_path, skipped):
            rows.append(row)
            if len(rows) - last_log >= 500:
                logging.info("[collect] %d rows kept so far (this source: %s)",
                             len(rows), src)
                last_log = len(rows)
        kept = len(rows) - n_before
        skip_by_source[src] = skipped
        logging.info("[%s] kept %d rows; skipped %d by reason: %s",
                     src, kept, sum(skipped.values()), dict(skipped))
    return rows, skip_by_source


def balanced_sample(
    rows: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Pick N rows balanced across sources, deterministic w.r.t. seed.

    Round-robin draws from per-source shuffled queues. Sources that run dry
    are dropped, remaining quota is filled from the others. If total < N,
    we return everything (and warn)."""
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for src, lst in by_source.items():
        rng.shuffle(lst)

    if sum(len(v) for v in by_source.values()) <= n:
        # Take everything, but still shuffle deterministically.
        all_rows = [r for lst in by_source.values() for r in lst]
        rng.shuffle(all_rows)
        return all_rows

    queues = {src: iter(lst) for src, lst in by_source.items()}
    picked: list[dict] = []
    src_order = list(queues.keys())
    rng.shuffle(src_order)
    while len(picked) < n and queues:
        for src in list(src_order):
            if src not in queues:
                continue
            try:
                picked.append(next(queues[src]))
            except StopIteration:
                queues.pop(src)
                src_order.remove(src)
                continue
            if len(picked) >= n:
                break
    return picked


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    tmp.rename(out_path)


def print_image_download_instructions(sources: list[str]) -> None:
    print()
    print("=" * 72)
    print("IMAGE ARCHIVES TO DOWNLOAD ON THE GPU BOX")
    print("=" * 72)
    print(
        "The JSONL stores image paths relative to the domain image dir.\n"
        "On the H100 box (NOT this Mac), pull the archives below and extract\n"
        "into a single root directory. Then pass that directory as\n"
        "  --images-root <root>\n"
        "to the downstream sampling script (see data/README.md).\n"
    )
    for src in sources:
        spec = SOURCES[src]
        print(f"# {src} -> images expected at <root>/{spec['images_dir']}/<img_filename>")
        for arc in spec["archives"]:
            print(f"  hf download OS-Copilot/OS-Atlas-data {arc} \\")
            print(f"      --repo-type dataset --local-dir <root>/_archives")
        domain_dir = Path(spec["images_dir"]).parent  # e.g. desktop_domain
        archive_basenames = [Path(a).name for a in spec["archives"]]
        if len(spec["archives"]) > 1:
            # Multi-part: cat parts back into the .zip then extract.
            # Final zip name follows README convention (replace "image" -> "images").
            joined_zip = archive_basenames[0].rsplit("_", 1)[0].replace("_image", "_images") + ".zip"
            cat_glob = f"<root>/_archives/{domain_dir}/{archive_basenames[0].rsplit('_', 1)[0]}_*"
            print(f"  cat {cat_glob} > <root>/_archives/{joined_zip}")
            zip_to_extract = f"<root>/_archives/{joined_zip}"
        else:
            zip_to_extract = f"<root>/_archives/{domain_dir}/{archive_basenames[0]}"
        print(f"  7z x {zip_to_extract} -o<root>/{domain_dir}/")
        print()


# ---------- main ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Pool size bumped 5k -> 50k for the streaming RSSD daemon: the sampler
    # runs continuously across multiple shards and consumes the pool linearly,
    # so a bigger pool = more headroom for an overnight run without re-feeding.
    # The collect() pass is still streaming (iter_filtered_elements is a
    # generator), so a 50k JSONL doesn't materialize the whole upstream JSON.
    p.add_argument("--n", type=int, default=50000, help="Number of items in the hard subset (default: 50000).")
    p.add_argument("--output", type=Path, default=Path("data/osatlas_hard_50k.jsonl"),
                   help="Output JSONL path (default: data/osatlas_hard_50k.jsonl).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0).")
    p.add_argument("--sources", type=str, default=DEFAULT_SOURCES,
                   help=f"Comma-separated source ids (default: {DEFAULT_SOURCES}). "
                        f"Choices: {','.join(SOURCES)}")
    p.add_argument("--cache-dir", type=Path, default=Path("data/.osatlas_cache"),
                   help="Where to cache downloaded annotation JSONs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise SystemExit(f"FATAL: unknown sources {unknown}. Choices: {list(SOURCES)}")

    logging.info("collecting from sources: %s", sources)
    logging.info("filter: bbox area < %.2f%%, instruction %d-%d words",
                 MAX_BBOX_AREA_FRAC * 100, MIN_WORDS, MAX_WORDS)

    rows, skip_by_source = collect(sources, args.cache_dir)
    logging.info("[collect] total kept across all sources: %d", len(rows))

    if len(rows) == 0:
        raise SystemExit(
            "FATAL: zero rows passed the filter. The annotation schema may have "
            "changed upstream. Re-inspect "
            "https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data and "
            "update SOURCES / iter_filtered_elements. Do NOT fall back to a "
            "synthetic dataset."
        )

    if len(rows) < args.n:
        logging.warning(
            "only %d rows passed the filter; less than --n=%d. Writing all of them. "
            "Consider adding more sources (e.g. --sources includes web/seeclick).",
            len(rows), args.n,
        )

    picked = balanced_sample(rows, args.n, args.seed)
    write_jsonl(picked, args.output)

    # Summary by source
    by_src = Counter(r["source"] for r in picked)
    logging.info("wrote %d rows to %s", len(picked), args.output)
    logging.info("rows per source in output: %s", dict(by_src))

    print_image_download_instructions(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
