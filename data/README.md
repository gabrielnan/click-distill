# `data/` — OS-Atlas hard subset prep

Scripts that build the **5k "hard" subset** of OS-Atlas grounding data used by
the RSSD lane (see `../PLAN.md`). Run on the H100 Brev box, not your laptop.

## Source

- Dataset: [`OS-Copilot/OS-Atlas-data`](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data)
- Paper: [OS-ATLAS: A Foundation Action Model for Generalist GUI Agents](https://arxiv.org/abs/2410.23218)
- License: **Apache-2.0** (see the dataset card for upstream-source citations
  for SeeClick / FineWeb / RICO / AMEX, etc. — they apply transitively).
- Total upstream size: ~800 GB. We download only ~270 MB of annotation JSONs
  here, then a subset of image archives (default ~75 GB for desktop images,
  see below).

## Schema (input)

Each annotation JSON (e.g. `desktop_domain/linux_splited.json`) is a top-level
list of:

```json
{
  "img_filename": "output_20240912_152854_original_screenshot.png",
  "elements": [
    {
      "instruction": "Save to Google Drive",
      "bbox": [0.50, 0.12, 0.61, 0.14],
      "data_type": "link"          // optional
    },
    ...
  ]
}
```

Bbox is **normalized `[left, top, right, bottom]`** with each value in `[0, 1]`
expressed as a fraction of image width / height.

## Schema (output JSONL)

`osatlas_hard_5k.jsonl` — one row per grounding element:

```json
{
  "image_path": "desktop_domain/windows_images/20240828_142303_before_screenshot.png",
  "instruction": "Log in to X / X",
  "bbox": [0.800, 0.000, 0.819, 0.024],
  "bbox_format": "normalized_xyxy",
  "image_width": null,
  "image_height": null,
  "source": "desktop/windows",
  "data_type": null
}
```

- `image_path` is **relative** to the `--images-root` you pass to the
  downstream sampling script. Joining gives the absolute PNG path.
- `bbox` stays normalized `[0,1]`. To convert to pixel xyxy at sampling time:
  `[l*W, t*H, r*W, b*H]` where `W,H = Image.open(path).size`.
- `image_width` / `image_height` are `null` and **filled by the consumer**
  when it loads the image. We do this lazily because the input dataset
  doesn't carry dims and probing 5k PNGs at prep time is wasteful when the
  sampling script will open every image anyway.

## Hard-subset filter

Per `PLAN.md`:

| Filter | Threshold | Rationale |
|---|---|---|
| bbox area | < 1% of screen area (`(r-l)*(b-t) < 0.01`) | "Hard" small-target clicks are where Northstar struggles |
| instruction length | 5–30 words | Drops 1-3 word labels ("OK", "Save") which are too easy / too ambiguous |
| domain | desktop (web optional via flag) | Non-mobile per plan |
| bbox sanity | `0 ≤ l < r ≤ 1`, `0 ≤ t < b ≤ 1`, all numeric | Rejects malformed annotations |

Empirical yield (per source, before sampling):

| Source | Annotation size | Elements | Pass filter |
|---|---|---|---|
| `desktop/linux` | 14 MB | 43k | ~2.1k |
| `desktop/macos` | 6.5 MB | 18k | ~0.6k |
| `desktop/windows` | 268 MB | 1.07M | ~163k |
| `web/seeclick` | 454 MB | (not measured here) | many |

Default `--sources desktop/linux,desktop/macos,desktop/windows` gives ~165k
candidates; the script round-robins per source and stops at `--n` (default
5000). Add `web/seeclick` if you want web in the mix.

## Run

```bash
# from repo root
pip install -r requirements.txt        # one-time
python data/download_osatlas.py        # writes data/osatlas_hard_5k.jsonl
```

CLI:

```
--n         number of items in the subset (default 5000)
--output    output path           (default data/osatlas_hard_5k.jsonl)
--seed      sampling seed         (default 0)
--sources   comma list of source ids
            choices: desktop/linux, desktop/macos, desktop/windows, web/seeclick
            default: desktop/linux,desktop/macos,desktop/windows
--cache-dir where to cache annotation JSONs (default data/.osatlas_cache)
```

The script:

1. Downloads the per-source annotation JSON to `data/.osatlas_cache/` (skipped
   if already cached). No image data is fetched at this stage.
2. Streams items, expands to per-element rows, applies the hard-subset filter.
3. Round-robin samples `--n` rows balanced across sources, deterministic
   under `--seed`.
4. Writes JSONL atomically (`*.partial` then `mv`).
5. Prints the `hf download` + `7z` commands needed to fetch the image
   archives that the JSONL refers to.

If the upstream schema changes or the dataset disappears, the script
**fails loudly with a `FATAL:` message** and a pointer to the dataset URL.
There is no synthetic fallback — that's a deliberate "be loud, don't lie"
choice for a hackathon dataset prep.

## Image archives — fetch separately

The JSONL only stores image *paths*. The PNGs live in big zip archives. On
the H100 box, after running the script above, follow the printed `hf download`
+ `7z x` instructions, or this canonical recipe for the default sources:

```bash
mkdir -p ~/osatlas_images/_archives
cd ~/osatlas_images/_archives

# Linux  (~3 GB compressed)
hf download OS-Copilot/OS-Atlas-data desktop_domain/linux_images.zip \
    --repo-type dataset --local-dir .
7z x desktop_domain/linux_images.zip -o../desktop_domain/

# macOS  (~5 GB compressed)
hf download OS-Copilot/OS-Atlas-data desktop_domain/macos_images.zip \
    --repo-type dataset --local-dir .
7z x desktop_domain/macos_images.zip -o../desktop_domain/

# Windows (~67 GB compressed, split into 4 parts)
for p in aa ab ac ad; do
    hf download OS-Copilot/OS-Atlas-data desktop_domain/windows_image_$p \
        --repo-type dataset --local-dir .
done
cat desktop_domain/windows_image_* > windows_images.zip
7z x windows_images.zip -o../desktop_domain/
```

Then pass `--images-root ~/osatlas_images` to the downstream sampling script;
`image_path` from the JSONL appended to that root resolves to a real PNG.

## Expected runtime + disk on H100

Annotation prep step (`download_osatlas.py`):

| Resource | Default (linux+macos+windows) | + `web/seeclick` |
|---|---|---|
| Network | ~290 MB download | ~750 MB |
| Disk (cache) | ~290 MB in `data/.osatlas_cache/` | ~750 MB |
| Disk (output) | ~1.5 MB JSONL | ~1.5 MB |
| Wall time | < 1 min on a Brev H100 | ~2 min |
| GPU | none (pure CPU + I/O) | none |
| RAM | < 1 GB peak (largest JSON ~270 MB loaded) | ~2 GB peak (seeclick ~454 MB) |

Image archive download (separate, run once on the H100):

| Source set | Compressed | Extracted | DL time @ 200 MB/s |
|---|---|---|---|
| linux + macos | ~8 GB | ~12 GB | ~40 s |
| + windows (default) | ~75 GB | ~110 GB | ~6 min |
| + seeclick web | ~195 GB | ~280 GB | ~16 min |

If you're tight on disk on the H100, drop windows and run with
`--sources desktop/linux,desktop/macos`; you'll get ~2.7k filtered rows
instead of 5k. Re-run with `--n 2700` to sample everything.

## Reproduction (canonical Brev command)

```bash
# from repo root, on the H100 Brev box
pip install -r requirements.txt
python data/download_osatlas.py \
    --n 5000 \
    --output data/osatlas_hard_5k.jsonl \
    --seed 0 \
    --sources desktop/linux,desktop/macos,desktop/windows
```
