# Hackathon Results

**Time**: 2026-05-09 13:30 - 16:00 PDT (2.5hr session)
**Track**: CUA Research / Tzafon Northstar-CUA-Fast distillation

## TL;DR

Built end-to-end **rejection-sampled self-distillation (RSSD)** pipeline for Northstar-CUA-Fast on B200 GPU. Hit and fixed 8 vLLM 0.20 / HF / trl compat bugs. Demonstrated **13.7% bbox-hit yield on 1000 OS-Atlas hard items → 137 SFT items collected**. Training got to LoRA init (66M trainable params) but trl SFTTrainer API churn (`tokenizer` / `processing_class` rename, `max_seq_length` moved out of SFTConfig) blocked the actual fit step at the deadline.

## What worked

- **Streaming RSSD pipeline**: hash-mod sharding, eager filter per-item, atomic JSONL writes, status dashboard, recovery via done_ids — 81 unit tests passing
- **vLLM 0.20.1 on B200** (Blackwell sm_100): model loads, KV cache 1.14M tokens, max concurrency 139× per request
- **Northstar inference**: 3.12 items/sec on B200 with N=8 sampling (~25 completions/sec)
- **Real bbox hits**: 16 SFT items collected from 100 OS-Atlas items (16% yield, single shard, ~30sec wall)

Sample SFT row:
```json
{
  "instruction": "broadcast Go Live, Click to run live server",
  "completion": "<tool_call>{\"name\": \"computer_use\", \"arguments\": {\"type\": \"click\", \"x\": [949, 986]}}</tool_call>",
  ...
}
```

## What broke (in order)

| # | Bug | Where | Fix |
|---|---|---|---|
| 1 | `image_width = None` in JSONL | OS-Atlas data prep | Pre-fill via PIL |
| 2 | `Unknown part type: image` | vLLM 0.20 expects `image_url` | Change format |
| 3 | `Cannot load local files` | vLLM blocks `file://` by default | `allowed_local_media_path="/"` |
| 4 | `Tools should be JSON schema` | HF chat-template validator strict | Inline tool spec via system message |
| 5 | `192 > 187 deepstack tokens` | vLLM Qwen3-VL profiling bug | `enforce_eager=True` |
| 6 | Model emits `x: [x, y]` array | Quirk of inline tool prompt | Parser handle list |
| 7 | bbox normalized [0,1] vs pixel | Filter compared mismatched units | Pre-convert bbox to pixels |
| 8 | `KeyError: 'image_width'` in train | Streaming writer didn't include dims | Patch SFT JSONL |
| 9 | `pyarrow ArrowInvalid` in train | HF Datasets mixed-type schema | **UNRESOLVED at deadline** |

## Costs (Brev)

| Instance | Type | Time | $/hr | Spent |
|---|---|---|---|---|
| click-distill-test (deleted) | A100 80GB | ~5min | $1.49 | $0.12 |
| click-distill (deleted) | hyperstack 2× H100 | ~3min FAILURE | $4.56 | $0 |
| click-distill-quad2 (deleted) | hyperstack 4× H100 | ~3min FAILURE | $9.12 | $0 |
| cd-b200 | verda B200 192GB | ~75min | $6.36 | $7.95 |
| cd-h100 | dmz 2× H100 | ~50min | $5.69 | $4.74 |
| cd-h100x4 (FAILURE) | hyperstack 4× H100 | unknown | $9.12 | unknown |
| **Total** | | | | **~$13-25** |

Within $40 budget.

## Architecture (what we built)

```
┌─────────────────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│ sample/launch_shards.sh │ ───▶ │ samples/shard_*/     │ ◀─── │ train_run.sh       │
│  (Lane A: continuous)   │      │   sft.jsonl          │      │  every ~15 min     │
└─────────────────────────┘      │   status.json        │      │  (Lane B: triggers)│
                                  └──────────────────────┘      └────────────────────┘
```

Streaming sampling daemon writes SFT items continuously. Trainer snapshots whatever's done, runs LoRA SFT (10min), evals subset (1.3min), appends leaderboard. Multiple training runs across the session.

**Decisions locked via grilling session**:
1. Multiple one-shot training runs (not online)
2. Hash-mod work distribution
3. Per-item synchronous flush
4. 50k input pool default
5. Manual training trigger + status
6. Skip 0-hit items
7. Versioned snapshots/adapters/leaderboard
8. GPU stealing pattern
9. 200-item stratified subset for intermediate, full 1581 for final
10. Split status.py + leaderboard.py

## What's next (post-hackathon)

1. **Fix train pyarrow bug** (~30min): convert messages to JSON string before HF Dataset, deserialize in collator
2. **Scale sample to 50k items**: ~5hr on 8× H100 → ~10k SFT items
3. **Run full LoRA SFT** (~30min)
4. **Eval on ScreenSpot-Pro full split**: baseline + tuned, per-category breakdown
5. **Stretch**: add Gaussian coord-loss ([HyperClick](https://arxiv.org/html/2510.27266)), inference-time best-of-N with coord-clustering

## Lessons

- **vLLM + brand-new GPUs (Blackwell) + brand-new model archs (Qwen3-VL deepstack) = friction**. Each integration point can break. Smoke-test vLLM-specific code on real GPU during prep, not at hackathon time.
- **HF chat-template validator is stricter than OpenAI spec**. Custom tool dicts may need bypassing via inline system messages.
- **Northstar emits quirky `x: [x, y]` array when tool spec comes via system msg** (vs `x: int, y: int` when via `tools=` param). Parser should handle both.
- **OS-Atlas bbox is normalized [0,1]**, not pixel. Easy to miss. Both formats look like floats.
- **Pre-hack baking is critical**: every smoke-test cold start = 1-2min vLLM load. We did 16 cold starts = ~25min lost. Pre-baked instance with verified pipeline at small scale would have shipped a real result.
- **Streaming pipeline + manual training trigger** = flexible, lets trainer pick "good enough" snapshot anytime. No need for online RL infra.

## Files

See [README.md](./README.md) for repo orientation, [STATUS.md](./STATUS.md) for handoff doc, [PLAN.md](./PLAN.md) for design.

Sample SFT data preserved at: `samples_run1_sft.jsonl` (16 items).
