# click-distill — Hackathon Plan (v2: 2hr × 2 people)

**Goal**: lift [Northstar-CUA-Fast](https://huggingface.co/Tzafon/Northstar-CUA-Fast) (4B) click acc on **[ScreenSpot-Pro](https://arxiv.org/abs/2504.07981)** by ≥3pp.

**Budget**: 2hr wall × 2 people. ~$10-15 of $40 [Brev](https://brev.dev) credits. 1× H100.

## Method: Pure RSSD (Rejection-Sampled Self-Distillation)

Drop all external teachers. Northstar samples N=8, filter by gt-bbox, LoRA SFT on hits.

**Why no teacher**: setup cost (download 32GB, format align) eats 30min — too much. Self-distill alone gives clean lift per [Tzafon's own ablation](https://www.tzafon.ai/blog/training-vlm-for-cua) (high-N pulls correct samples from model's tail).

```
ScreenSpot-Pro train + OS-Atlas hard subset (5k items)
                    │
            Northstar N=8 sample
                    │
        filter: click ∈ gt_bbox (yield ~3k)
                    │
            LoRA SFT (1 epoch)
                    │
        eval: ScreenSpot-Pro full
```

## Models + data

| Item | Source | Size |
|---|---|---|
| Student | [Northstar-CUA-Fast](https://huggingface.co/Tzafon/Northstar-CUA-Fast) | 4B |
| Data pool | [OS-Atlas data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) hard subset (small bbox, web/desktop) | 5k items |
| Eval | [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) full test | 1.5k |
| Sanity | [ScreenSpot-v2](https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2) (skip if tight) | 1.2k |

## Two-lane parallel work

### Lane A — Data + Sampling (Person 1)
| t | task |
|---|------|
| 0:00–0:15 | Brev H100 up. Clone [lightcone](https://github.com/tzafon/lightcone). Download Northstar weights, OS-Atlas subset (5k), ScreenSpot-Pro test |
| 0:15–0:45 | vLLM serve Northstar. Sample N=8 × 5k = 40k completions. Temp 0.7 |
| 0:45–1:00 | Filter: parse `(x,y)`, check ∈ gt_bbox normalized 0-999. Save JSONL of hits |
| 1:00–1:15 | Hand off SFT data to Lane B. Start baseline eval (Northstar greedy on ScreenSpot-Pro) |
| 1:15–2:00 | Monitor training, run tuned eval, plot results |

### Lane B — Train + Eval (Person 2)
| t | task |
|---|------|
| 0:00–0:30 | Set up [unsloth](https://github.com/unslothai/unsloth) env. Write `train.py` (LoRA r=32, lr=1e-4, 1 epoch) |
| 0:30–1:00 | Write `eval.py` for ScreenSpot-Pro. Test on small sample with raw Northstar |
| 1:00–1:15 | Receive data from Lane A. Sanity-check format (~10 examples) |
| 1:15–1:45 | Launch LoRA SFT (~30min for 3k items, 1 epoch, bs=4 grad-accum=8) |
| 1:45–2:00 | Eval tuned model. Per-app-category breakdown |

## Hyperparams (no time to sweep — set + ship)

```python
# LoRA
r = 32
alpha = 64
target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
freeze_vision = True

# Training
batch_size = 4
grad_accum = 8     # eff bs = 32
lr = 1e-4
schedule = "cosine"
warmup_ratio = 0.05
epochs = 1
precision = "bf16"

# Sampling
N = 8
temp = 0.7
top_p = 0.9
```

## Eval format

```python
# ScreenSpot-Pro item: {image, instruction, bbox: [x1,y1,x2,y2]}
# Northstar output: action with click(x, y) where x,y ∈ [0, 999]
def hit(pred_xy, gt_bbox, img_w, img_h):
    x = pred_xy[0] * img_w / 1000
    y = pred_xy[1] * img_h / 1000
    return gt_bbox[0] <= x <= gt_bbox[2] and gt_bbox[1] <= y <= gt_bbox[3]
```

## Deliverables (post-hack writeup)

1. `results.json`: baseline vs tuned, overall + per-category
2. Plot: `acc_per_category.png`
3. LoRA adapter pushed to HF: `gabrielnan/Northstar-CUA-Fast-rssd-lora`
4. README with reproduction command

## Risks (compressed)

| Risk | Mitigation |
|---|---|
| Brev H100 cold-start slow | Pre-warm before hackathon kickoff |
| Yield <1k after filter | Drop temp to 0.5, increase N to 16 (+15min) |
| Training diverges | Drop lr to 5e-5, fall back r=16 |
| Eval format mismatch | Lane B builds eval against raw Northstar in hour 0:30–1:00 — catches this early |
| Data leakage (OS-Atlas overlaps ScreenSpot-Pro source images) | Spot-check 20 items, dedupe by perceptual hash if needed |

## Cut from previous plan

- ~~Qwen3-VL-32B teacher~~ (32GB download eats 15min; uncertain lift)
- ~~UI-TARS-1.5-7B teacher~~ (format alignment eats 20min)
- ~~Claude rationale generation~~ (cute but optional, cut for time)
- ~~50k item subset~~ (cut to 5k for sampling speed)
- ~~OSWorld stretch eval~~ (skip; ScreenSpot-Pro is the headline)
- ~~ScreenSpot-v2 sanity~~ (run only if time at end)

## Stretch (only if Lane B finishes by 1:30)

Inference-time best-of-N w/ coord-cluster on tuned model — see [I33](./IDEAS.md#i33). Adds 5min, likely +2pp.

## Pre-hack checklist (do night before, not during the 2hr)

- [ ] Brev account with $40 credits confirmed
- [ ] H100 instance template ready (CUDA 12.4, vLLM, unsloth, transformers, peft, trl pre-installed)
- [ ] HF auth token in env
- [ ] Northstar-CUA-Fast weights cached (~8GB)
- [ ] OS-Atlas hard subset (5k) pre-downloaded as JSONL
- [ ] ScreenSpot-Pro test split downloaded
- [ ] Eval script tested end-to-end on 10 items
- [ ] Repo cloned to instance, branches per-person

**This is the highest-leverage prep.** Without it, you lose 30min of the 2hr to setup.
