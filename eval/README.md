# click-distill eval/

Click-grounding evaluation of [Northstar-CUA-Fast](https://huggingface.co/Tzafon/Northstar-CUA-Fast)
on [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) (1.5k high-resolution
professional-app GUI grounding tasks). Used both for the baseline number we're
trying to beat and for measuring the LoRA-tuned model.

## Dataset

- HF: <https://huggingface.co/datasets/likaixin/ScreenSpot-Pro>
- Upstream eval ref: <https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding>
- Paper: <https://arxiv.org/abs/2504.07981>
- Size: ~1,581 items, ~3.4 GB (images dominate)
- License: MIT

### Schema (per-item, after `download_screenspot_pro.py` flattens it)

```json
{
  "id": "vscode_macos_0",
  "image": "images/vscode_mac/screenshot_2024-12-03_15-15-02.png",
  "instruction": "Refresh the file explorer.",
  "instruction_cn": "刷新文件资源管理器。",
  "bbox": [473, 183, 503, 219],
  "img_size": [2560, 1664],
  "application": "vscode",
  "platform": "macos",
  "group": "Dev",
  "ui_type": "icon",
  "task_filename": "vscode_macos"
}
```

- `bbox` is `[x1, y1, x2, y2]` in **source-image pixels**.
- `img_size` is `[width, height]`.
- `group` is one of `Dev`, `Creative`, `CAD`, `Scientific`, `Office`, `OS`.
- `ui_type` is `text` or `icon` (icons are systematically harder).
- 27 per-application annotation files; we concatenate into one JSONL.

The dataset has only one effective split (the test set). The HF repo ships
positive samples only here; negative samples (where there is no valid target)
are part of the upstream `eval_screenspot_pro_parallel.py` flow and we skip
them — click accuracy on positive samples is the headline metric.

## Northstar action format

Northstar-CUA-Fast is a 4B Qwen3-VL-based CUA model. For grounding, it expects
a `computer_use` tool spec in the system prompt and emits a `<tool_call>` block:

```text
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [423, 567]}}
</tool_call>
```

- **Coordinate space: 0–1000 normalised** (Tzafon's lightcone SDK
  `examples/coordinate_scaling.py` confirms this; the system prompt explicitly
  tells the model "the screen's resolution is 1000x1000").
- Convert to pixels: `x_px = x_norm * img_w / 1000`, same for y.
- A click is correct iff the pixel point lies inside the GT bbox.

`parse_action.py` handles malformed outputs gracefully (bare JSON, 4-coord
bboxes collapsed to centres, last-resort regex). Run `python eval/parse_action.py`
to verify the parser on synthetic cases — it's pure-Python, no GPU needed.

## Image preprocessing

We use Qwen3-VL's `smart_resize` with `factor=32`, `min_pixels=32*32`,
`max_pixels=12_845_056` (≈12.8M, matches the reference qwen3vl.py in the
ScreenSpot-Pro repo). vLLM applies this via `mm_processor_kwargs`.

ScreenSpot-Pro images are huge (often 2.5k × 1.6k or larger) — that's the
whole point of the benchmark. Don't lower `--max-pixels` without checking
that small icons remain resolvable; the benchmark targets are often <50 px.

## Reproduction

```bash
# 1. Install deps (from repo root)
pip install -r requirements.txt

# 2. Download the dataset (~3.4 GB, idempotent)
python eval/download_screenspot_pro.py

# 3. Sanity-check the pipeline on 10 items first (≈30 s on H100)
python eval/eval_screenspot_pro.py \
    --model Tzafon/Northstar-CUA-Fast --limit 10

# 4. Full baseline run (~10 min on 1xH100 with vLLM batching)
python eval/eval_screenspot_pro.py \
    --model Tzafon/Northstar-CUA-Fast \
    --output eval/results/baseline.json

# 5. Tuned-model run (after Lane B finishes LoRA SFT)
python eval/eval_screenspot_pro.py \
    --model Tzafon/Northstar-CUA-Fast \
    --lora ./outputs/rssd-lora \
    --output eval/results/tuned.json
```

Headline number to read off `results.json`: `metrics.overall.acc` and the
`metrics.by_group` breakdown. Compare baseline vs. tuned per group — that's
the deliverable plot.

## Expected baseline numbers

**Tzafon has not published a Northstar-CUA-Fast number on ScreenSpot-Pro.**
Their blog reports OS-World Chrome lifts (≈+20 pp), not ScreenSpot-Pro.

For reference (other 4B-class GUI-grounding models on ScreenSpot-Pro overall
positive acc): UI-TARS-7B ≈18 %, OS-Atlas-7B ≈18 %, Qwen2.5-VL-7B-Instruct
≈22 %, top SOTA ≈40-50 %. A 4B specialised CUA model in the 15-30 % range is
plausible; we'll know the real number after step 4 above.

**Treat the baseline as unknown until measured.** The +3 pp goal in
`PLAN.md` is over our own measured baseline, not a published one.

## Output JSON

`results.json` payload:

```json
{
  "model": "Tzafon/Northstar-CUA-Fast",
  "lora": null,
  "n_items": 1581,
  "elapsed_sec": 612.4,
  "metrics": {
    "overall": {"n": 1581, "acc": 0.231, "wrong_format": 12},
    "by_group": {"CAD": {"n": 240, "acc": 0.12, "wrong_format": 3}, ...},
    "by_application": {...},
    "by_ui_type": {"text": {...}, "icon": {...}},
    "by_platform": {...}
  },
  "predictions": [
    {"id": "...", "raw_response": "...", "pred_norm": [423, 567],
     "pred_pixel": [1083.2, 944.0], "correct": true, "wrong_format": false,
     "bbox": [...], "instruction": "...", ...},
    ...
  ]
}
```

`predictions` is the full per-item dump — keep it for later analysis (failure
clustering, coordinate-distribution plots, etc.).

## Backends

- `--backend vllm` (default): batched, fast. Required for the full run to
  fit in the 2-hour budget. Uses `LLM.chat()` with inline images.
- `--backend transformers`: one image at a time via `AutoModelForImageTextToText`
  + optional `peft.PeftModel` overlay. ~50× slower; use only if vLLM breaks.

## Gotchas

1. **Coordinate space is 0–1000, not 0–999.** The blog says "0-999"; the
   actual prompt string says `1000x1000`. Off-by-one doesn't matter for the
   normalised → pixel conversion, but use `* img_w / 1000` consistently.
2. **`bbox` is xyxy not xywh.** Easy to flip if you're used to COCO.
3. **`img_size` is `[w, h]`** — width first. Don't swap.
4. **Images are large**; `max_pixels` controls vLLM memory. If you OOM on a
   single H100, lower `--max-num-seqs` (try 8) before lowering `--max-pixels`.
5. **`ui_type=icon` items are dramatically harder than `text` items**; report
   both numbers in writeups — overall hides the icon failure mode.
6. **vLLM LoRA**: needs `enable_lora=True` and `--max-lora-rank` ≥ trained
   rank (we use 64 to leave headroom over PLAN.md's r=32).
