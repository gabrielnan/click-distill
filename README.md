# click-distill

Improve [Northstar-CUA-Fast](https://huggingface.co/Tzafon/Northstar-CUA-Fast) (4B Qwen3-VL CUA model) click accuracy on [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) via **rejection-sampled self-distillation (RSSD)**.

Tzafon CUA hackathon — research track. 2hr × 2 people, $40 [Brev](https://brev.dev) credits.

## What

Pipeline:
1. Sample N=8 completions per item from Northstar on a hard subset of OS-Atlas (5k items)
2. Filter: keep completions whose `(x, y)` click lands in the ground-truth bbox
3. LoRA SFT Northstar on the filtered (image, instruction, completion) triples
4. Eval click accuracy on ScreenSpot-Pro

Why this works: see [PLAN.md](./PLAN.md) §"Method" and the discussion in chat — RSSD converts test-time best-of-N gains into amortized greedy gains, while preserving Northstar's RL-tuned action format (unlike off-policy SFT on synthetic labels).

## Repo layout

```
click-distill/
├── IDEAS.md                  # 65 ranked hackathon idea pool
├── PLAN.md                   # current execution plan (v2, 2hr × 2 people)
├── northstar_format.py       # SHARED: tool spec + action parser (single source of truth)
├── train_run.sh              # one-shot orchestrator: kill→freeze→train→eval→leaderboard→relaunch
├── data/
│   ├── download_osatlas.py   # OS-Atlas hard-subset (5k) builder
│   └── README.md
├── sample/
│   ├── sample_northstar.py   # vLLM sampler + StreamingWriter (per-shard sft.jsonl)
│   ├── launch_shards.sh      # spawn N sampler daemons (one per GPU)
│   ├── kill_shard.sh         # stop a specific shard (used by train_run.sh)
│   ├── status.py             # dashboard over samples/shard_*/status.json
│   ├── filter_by_bbox.py     # legacy non-streaming filter
│   └── README.md
├── train/
│   ├── freeze_snapshot.py    # samples/shard_*/sft.jsonl  ->  snapshots/run_<id>/
│   ├── train_lora.py         # LoRA SFT on a snapshot, writes adapters/lora_<id>/
│   └── README.md
├── eval/
│   ├── download_screenspot_pro.py
│   ├── eval_screenspot_pro.py     # vLLM eval; --subset N for stratified slice
│   ├── stratified_subset.py       # deterministic 200-item subset builder
│   ├── leaderboard.py             # results/leaderboard.{txt,json}
│   ├── finalize.py                # full 1581-item eval on best adapter
│   ├── parse_action.py            # legacy shim → northstar_format
│   └── README.md
├── tests/                    # pytest, no GPU/vLLM required
└── requirements.txt
```

## Streaming RSSD pipeline

The hackathon uses a streaming setup: Lane A's sampler daemons write SFT
items continuously while Lane B's trainer runs multiple LoRA training runs
on growing snapshots of that data.

```
┌─────────────────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│ sample/launch_shards.sh │ ───▶ │ samples/shard_*/     │ ◀─── │ train_run.sh       │
│  (Lane A: continuous)   │      │   sft.jsonl          │      │  every ~15 min     │
└─────────────────────────┘      │   status.json        │      │  (Lane B: triggers)│
                                  └──────────────────────┘      └────────────────────┘
                                                                          │
                                  ┌──────────────────────┐                ▼
                                  │ snapshots/run_<id>/  │ ◀─── freeze_snapshot.py
                                  │   sft.jsonl          │                │
                                  │   snapshot_meta.json │                ▼
                                  └──────────────────────┘         train_lora.py
                                                                          │
                                                                          ▼
                                                                   adapters/lora_<id>/
                                                                          │
                                                                          ▼
                                                          eval_screenspot_pro.py --subset 200
                                                                          │
                                                                          ▼
                                                                  results/lora_<id>.json
                                                                  results/leaderboard.{txt,json}
```

Each `train_run.sh` invocation: kills sampler shard 0 to free GPU 0 → freezes
the snapshot → trains a fresh LoRA from base on it → subset-evals (200 stratified
items, ~1.3 min on H100) → appends the leaderboard → restarts sampler shard 0.

At the end (after the last training run), `python eval/finalize.py` reads
the leaderboard, picks the top-ranked adapter, and runs the FULL 1581-item
eval on it — this is the headline number for the writeup.

## Run order (Brev H100, after pre-hack setup)

```bash
# Lane A — Person 1: kick off the streaming sampler (all 4 shards)
python data/download_osatlas.py --n 5000 --output data/osatlas_hard_5k.jsonl
sample/launch_shards.sh                     # 4 daemons, one per GPU
# leave running; monitor with: python sample/status.py --watch

# Lane B — Person 2: download eval set, baseline, then per-tick training runs
python eval/download_screenspot_pro.py
python eval/eval_screenspot_pro.py --model Tzafon/Northstar-CUA-Fast \
    --subset 200 --output results/baseline.json
python -c "from eval.leaderboard import append_result; from pathlib import Path; \
    import json; p=json.loads(Path('results/baseline.json').read_text()); \
    append_result(Path('results'), 'baseline', { \
      'overall_acc': p['metrics']['overall']['acc'], \
      'text_acc': (p['metrics']['by_ui_type'].get('text') or {}).get('acc', 0.0), \
      'icon_acc': (p['metrics']['by_ui_type'].get('icon') or {}).get('acc', 0.0), \
      'subset_size': p['subset_size']})"

# Trigger a training run every ~15 min while the sampler accumulates more data:
./train_run.sh 1130
./train_run.sh 1145
./train_run.sh 1200
# ... etc.

# Finalize: full 1581-item eval on the best adapter.
python eval/finalize.py
cat results/final.json
```

See [train/README.md](./train/README.md) for the locked hyperparameters and
the snapshot/training contract; [sample/README.md](./sample/README.md) for
the streaming-shard layout and dashboard.

## Key references

- [Northstar-CUA-Fast model card](https://huggingface.co/Tzafon/Northstar-CUA-Fast)
- [Tzafon training-VLM-for-CUA blog](https://www.tzafon.ai/blog/training-vlm-for-cua) — explains the SFT plateau + positional encoding insights
- [Lightcone agent SDK](https://github.com/tzafon/lightcone) — source-of-truth for action format (`_cua.py` COORD_KEYS)
- [STaR paper](https://arxiv.org/abs/2203.14465) — original rejection-sampled self-distillation
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981) — eval benchmark

## Pre-hack checklist

See [PLAN.md §"Pre-hack checklist"](./PLAN.md#pre-hack-checklist-do-night-before-not-during-the-2hr) — without overnight setup you lose 30min of 2hr to env friction.
