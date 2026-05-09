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
├── data/
│   ├── download_osatlas.py   # OS-Atlas hard-subset (5k) builder
│   └── README.md
├── sample/
│   ├── sample_northstar.py   # vLLM sampler, N=8, temp=0.7
│   ├── filter_by_bbox.py     # parse + bbox filter, idempotent
│   └── README.md
├── eval/
│   ├── download_screenspot_pro.py
│   ├── eval_screenspot_pro.py # vLLM eval, per-category breakdown
│   ├── parse_action.py        # legacy shim → northstar_format
│   └── README.md
└── requirements.txt
```

## Run order (Brev H100, after pre-hack setup)

```bash
# Lane A — Person 1 (data + sampling + filter)
python data/download_osatlas.py --n 5000 --output data/osatlas_hard_5k.jsonl
# (then download image archives per data/README.md)
python sample/sample_northstar.py \
    --input data/osatlas_hard_5k.jsonl \
    --output sample/raw_samples.jsonl \
    --n 8 --temperature 0.7 --batch-size 64
python sample/filter_by_bbox.py \
    --input sample/raw_samples.jsonl \
    --output sample/sft_data.jsonl

# Lane B — Person 2 (eval baseline + train + eval tuned)
python eval/download_screenspot_pro.py
python eval/eval_screenspot_pro.py --model Tzafon/Northstar-CUA-Fast --output results/baseline.json
# wait for sft_data.jsonl from Lane A
python train/lora_sft.py --data sample/sft_data.jsonl --output adapters/rssd-lora     # TODO Lane B
python eval/eval_screenspot_pro.py --model Tzafon/Northstar-CUA-Fast \
    --lora adapters/rssd-lora --output results/tuned.json
```

## Key references

- [Northstar-CUA-Fast model card](https://huggingface.co/Tzafon/Northstar-CUA-Fast)
- [Tzafon training-VLM-for-CUA blog](https://www.tzafon.ai/blog/training-vlm-for-cua) — explains the SFT plateau + positional encoding insights
- [Lightcone agent SDK](https://github.com/tzafon/lightcone) — source-of-truth for action format (`_cua.py` COORD_KEYS)
- [STaR paper](https://arxiv.org/abs/2203.14465) — original rejection-sampled self-distillation
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981) — eval benchmark

## Pre-hack checklist

See [PLAN.md §"Pre-hack checklist"](./PLAN.md#pre-hack-checklist-do-night-before-not-during-the-2hr) — without overnight setup you lose 30min of 2hr to env friction.
