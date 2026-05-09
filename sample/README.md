# Rejection-sampled self-distillation pipeline

Sample N completions per item from **Northstar-CUA-Fast**, filter by
ground-truth bbox, emit SFT-ready JSONL. Lane B (training) consumes
`sft.jsonl` (streaming) or `sft_data.jsonl` (legacy).

Two modes:

* **Streaming shard mode** (default for the daemon): one worker per GPU,
  hash-mod input distribution, eager filter, per-item synchronous flush.
  Used by `sample/launch_shards.sh` + monitored via `sample/status.py`.
* **Legacy raw mode**: single GPU, all 8 raw completions written to
  `raw_samples.jsonl`, then `filter_by_bbox.py` produces `sft_data.jsonl`.
  Kept for back-compat / one-off runs.

## Streaming shard mode (recommended)

```bash
# 1. Build a 50k pool (one-shot, fast — only annotation JSONs are pulled).
python data/download_osatlas.py            # default: --n 50000

# 2. Launch one shard per GPU. Daemon-style: nohup + & per shard.
sample/launch_shards.sh \
    --num-shards 4 --gpus 0,1,2,3 \
    --input data/osatlas_hard_50k.jsonl \
    --samples-root samples

# 3. Watch progress (one-shot or --watch 5 for refresh).
python sample/status.py --samples-root samples
python sample/status.py --watch 5

# 4. Kill a shard or all of them.
sample/kill_shard.sh 0
sample/kill_shard.sh all
```

### Per-shard on-disk layout

```
samples/shard_0/
  sft.jsonl       # winning completions only (one per item that hit at least once)
  done_ids.txt    # one item_id per line, append-only
  zero_hits.txt   # item_ids that got 0/8 hits — permanently skipped on resume
  status.json     # {pid, gpu_id, started_at, last_write_at, items_done, sft_items, ...}
  pid             # for sample/kill_shard.sh
  log.txt         # stdout/stderr from the worker
samples/shard_1/...
```

Per-item write order (single-writer, crash-safe):

1. Append SFT row to `sft.jsonl`, `flush + fsync`.
2. Append item_id to `done_ids.txt`, `flush + fsync`.
3. Update `status.json` (atomic tmp + rename).

Crash between (1) and (2) → the SFT row is on disk but `done_ids.txt` doesn't
reference it. On resume the item is re-processed; the trainer dedupes by
`item_id`. Crash after (2) → invisible to consumers.

### Resume

`StreamingWriter.load_skip_set()` reads `done_ids.txt ∪ zero_hits.txt` at
launch and filters the input stream. `--no-resume` ignores prior state (rare —
useful only for debugging a stuck shard). Hash-mod and resume compose: the
shard's input is `{lines where idx % N == shard_id} \ skip`.

### Work distribution

Hash-mod: shard X processes input lines where `line_index % num_shards == X`.
No coordinator, no claims, no IPC. Two shards reading the same input file
cannot pick the same item.

## Legacy raw mode (single GPU)

```bash
# Sample (GPU required: 1x H100 80GB)
python sample/sample_northstar.py \
    --input  data/osatlas_hard_5k.jsonl \
    --output sample/raw_samples.jsonl \
    --n 8 --temperature 0.7 --top-p 0.9 \
    --max-tokens 512 --batch-size 64

# Filter (CPU only, ~30 sec, idempotent)
python sample/filter_by_bbox.py \
    --input  sample/raw_samples.jsonl \
    --output sample/sft_data.jsonl \
    --keep first
```

`--keep first` (default) writes one record per item — first sample whose click
lands in the gt bbox. Prevents any single screenshot from dominating SFT.
Use `--keep all` if Lane B wants diversity-aware training.

## Status dashboard format

```
shards: 4/4 alive (last write 8s ago)
items_done: 9234 (rate: 7.2/sec, +432/min)
sft_items: 6012 (yield: 65.1%)
elapsed: 22min
projected at T+45min: ~18000 items, ~11700 SFT
GPUs: gpu0=84% gpu1=81% gpu2=88% gpu3=82%
--------------------------------------------------------------
shard_0: 2308 done, 1521 sft (last write 7s ago)  alive
shard_1: 2295 done, 1487 sft (last write 12s ago) alive
shard_2: 2316 done, 1502 sft (last write 5s ago)  alive
shard_3: 2315 done, 1502 sft (last write 9s ago)  alive
```

If a shard's `last_write_at` is older than 60s, its row is marked **STALE**
(red on a TTY). The aggregate `alive` count drops accordingly.

## Tests

```bash
pytest tests/test_shard_filter.py tests/test_streaming_writer.py \
       tests/test_resume.py tests/test_launch_shards.py \
       tests/test_status.py tests/test_download_pool.py
```

All tests run on CPU. vLLM is imported lazily inside `main()` so the script
loads on a Mac for testing/help.

## Confirmed Northstar format (sources)

- **Base model**: Qwen3-VL-4B (`Qwen3VLForConditionalGeneration`,
  `model_type: qwen3_vl`). Fine-tuned with GRPO. Source:
  [config.json on HF](https://huggingface.co/Tzafon/Northstar-CUA-Fast/blob/main/config.json).
- **Chat template**: standard Qwen3-VL `<|im_start|>...<|im_end|>` with a
  `<tools>...</tools>` system block when `tools=[...]` is passed. Source:
  [tokenizer_config.json on HF](https://huggingface.co/Tzafon/Northstar-CUA-Fast/blob/main/tokenizer_config.json).
- **Tool**: a single `computer_use` function. Action types: `click`,
  `double_click`, `triple_click`, `right_click`, `drag`, `type`, `key`,
  `scroll`, `hscroll`, `navigate`, `wait`, `terminate`. Source:
  [lightcone/examples/_cua.py](https://github.com/tzafon/lightcone/blob/main/examples/_cua.py).
- **Coordinates**: model emits `x, y` as integers in **[0, 999] normalized**;
  client denormalizes via `x_pixel = x / 1000 * display_width`. Source:
  [`COORD_KEYS` in `_cua.py`](https://github.com/tzafon/lightcone/blob/main/examples/_cua.py)
  + [`coordinate_scaling.py` docstring](https://github.com/tzafon/lightcone/blob/main/examples/coordinate_scaling.py)
  ("Northstar outputs coordinates in a 0-1000 space; your code converts them
  to pixel coordinates before clicking").
- **Output shape**: `<tool_call>{"name":"computer_use","arguments":{...}}</tool_call>`
  emitted by the assistant turn (Qwen3-VL tool-call format from the chat
  template). Free-form text may precede the `<tool_call>`; the lightcone
  examples treat any pre-text as `OutputResponseOutputMessage` (rationale).
- **Image placement**: image first, text after, in the user message — matches
  both the model card snippet and `coordinate_scaling.py`.

## What we send per item

```python
tools = [{
  "type": "function",
  "function": {
    "name": "computer_use",
    "parameters": { "type": "object",
                    "properties": {"type": {...}, "x": {...}, "y": {...}, ...},
                    "required": ["type"] },
    # hint fields rendered into the tools JSON so the model sees the dims:
    "display_width":  <image_width>,
    "display_height": <image_height>,
    "environment": "desktop",
  }
}]
messages = [{"role": "user", "content": [
    {"type": "image", "image": "<abs path>"},
    {"type": "text",  "text":  "<instruction>"},
]}]
```

`vllm.LLM.chat(messages, sampling_params, tools=...)` applies the tokenizer's
chat template (which is the Qwen3-VL one shipped on HF), inlines the tool
schema into the system prompt, and handles image loading.

## Expected runtime (1x H100 80GB)

| stage | items | completions | wall time |
|---|---|---|---|
| sample (`vllm.chat`, n=8, batch=64, prefix-caching) | 5,000 | 40,000 | **~30 min** |
| filter (regex + json) | 5,000 | 40,000 | ~30 sec |

Numbers assume bf16, max_model_len=8192, max_image_tokens=2048
(`mm_processor_kwargs.max_pixels=2048*32*32=2M px`). Prefix caching shares
the system prompt across all items; per-image prefill dominates. If items
are processed at ~3/s, 5k items ~= 28 min.

## Expected yield

Theoretical ceiling: with Northstar baseline pass@1 ~= 35-45% on
ScreenSpot-style targets and N=8 independent samples,
`P(>=1 hit) = 1 - (1 - 0.4)^8 ~= 98%`. In practice expect **60-80% items
kept** because:

- Format errors: model occasionally emits malformed JSON (parse failure).
- Hard items: small bboxes + crowded UIs in OS-Atlas hard subset push some
  items below sampling reach.
- Coordinate noise: model can off-by-a-few-percent even when "right".

The filter prints exact stats. Recovery levers (per PLAN.md risks): if
yield <1k, drop temperature to 0.5 and bump N to 16.

## Re-running cheaply

`sample_northstar.py` resumes from existing output: it reads `item_id`s
already in `--output` and skips them. Append-only writes. Crash-safe.

`filter_by_bbox.py` rewrites `--output` from scratch every run (no GPU,
~30 s) — change `--keep` or add new parser fallbacks and re-run freely.

## Parsing notes

`filter_by_bbox.py` tries 4 parsers in order:

1. `<tool_call>{...}</tool_call>` — canonical, expected on >95% of well-formed samples.
2. Bare `{"name":"...","arguments":{...}}` JSON without XML wrap.
3. Bare action JSON `{"type":"click","x":..,"y":..}` (xy or yx order).
4. `click(x, y)` / `pyautogui.click(x=..., y=...)` — pretrain-style fallback.

Run `python sample/filter_by_bbox.py --self-test` to sanity-check the regexes
without any data. Use `--verbose-misses` to dump every parse-fail / bbox-miss
when tuning patterns.

## Gotchas

- **Multi-turn loop**: Northstar is *trained* as a multi-turn agent (see
  `lightcone/examples/_cua.py: run_cua_loop` — feeds `computer_call_output`
  back as new images). For SFT distillation we use **single-turn** sampling:
  one screenshot, one action. This matches how ScreenSpot-Pro is evaluated
  and how OS-Atlas data is shaped (single-step click instructions).
- **Tool list affects the system prompt**: passing `tools=[...]` triggers the
  `if tools` branch in the chat template, which prepends a
  `# Tools` block. The model has been RL-trained inside this format, so
  *always* pass the tool — sampling without it will drift to plain-text
  output and tank yield.
- **`display_width` / `display_height`**: not part of the standard OpenAI
  function schema, but Qwen3-VL's chat template `tojson`-dumps the entire
  tool dict verbatim, so any extra hint fields land in the system prompt.
  We pass per-item dims (the source image's actual `image_width` /
  `image_height`) so the 0-999 normalization the model emits maps cleanly
  back to pixels at filter time.
- **Image preprocessing**: Qwen3-VL uses `Qwen2VLImageProcessorFast` with
  `patch_size=16`, `merge_size=2`. We cap visual tokens to
  `max_image_tokens=2048` (~2M pixels) via `mm_processor_kwargs.max_pixels`.
  ScreenSpot-Pro / OS-Atlas screenshots can be large (1920x1080+), and at
  full res a single image can blow past 8k vision tokens — capping keeps
  prefill cheap without measurably hurting click accuracy on UI-scale
  targets.
- **vLLM tool support**: `LLM.chat(tools=[...])` requires recent vLLM
  (`>=0.6.3` for proper tool-call formatting via tokenizer chat template).
  Pinned in `requirements.txt`.
- **No `enforce_eager`**: cudagraph-based decoding is faster on H100;
  Qwen3-VL is supported. If vLLM throws, fall back to `enforce_eager=True`.
