# STATUS / HANDOFF — click-distill

**For**: princesavsaviya (H100 instance, spinning up)
**From**: gabriel (B200, ~14:55 PDT 2026-05-09)
**Deadline**: 16:00 PDT (~65 min remaining)
**Goal**: Improve Northstar-CUA-Fast (4B Qwen3-VL) click accuracy on ScreenSpot-Pro via rejection-sampled self-distillation (RSSD).

Repo: https://github.com/gabrielnan/click-distill (all code pushed, 81 tests passing)

---

## TL;DR — What you do

1. SSH your H100, clone repo, install deps (commands below)
2. Smoke-test sampling on 100 items
3. Launch sampling shards (use all GPUs, hash-mod sharded)
4. When ~5-10k SFT examples accumulated, train LoRA, eval on 200-item subset
5. If adapter beats baseline, run full 1581-item eval
6. Commit results

**You and I run independently.** Whichever finishes a good adapter first wins. Optionally we merge SFT JSONLs at the end (rsync/S3) — for now just race.

**DO NOT change hyperparameters.** All locked + tested. See PLAN.md for rationale.

---

## Read these files first (in order)

| # | File | Why |
|---|------|-----|
| 1 | `README.md` | Repo orientation |
| 2 | `PLAN.md` | Execution plan |
| 3 | `northstar_format.py` | **CRITICAL** — model action format, shared by sample + eval |
| 4 | `instance_setup.sh` | Brev setup script (read, don't blindly run — see below) |
| 5 | `IDEAS.md` | Full research context (65 ranked ideas) — only if you want context |

---

## Setup commands (your H100)

Ubuntu 24.04 enforces PEP 668 — must use `--break-system-packages`.

```bash
# 1. Auth + clone
gh auth login                    # browser flow, or paste token
git clone https://github.com/gabrielnan/click-distill.git ~/click-distill
cd ~/click-distill

# 2. Install (anonymous HF, public repos — no HF token needed)
pip install --break-system-packages -r requirements.txt
pip install --break-system-packages vllm
sudo apt install -y p7zip-full   # for OS-Atlas extraction

# 3. Verify CUDA + vLLM
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
python -c "import vllm; print(vllm.__version__)"   # should be 0.20.1+

# 4. Sanity-check tests pass (no GPU needed)
pytest tests/ -q

# 5. Download artifacts (run in parallel in tmux/background)
python data/download_osatlas.py --n 50000        # input image pool
# Northstar-CUA-Fast model auto-downloads on first vLLM load
# ScreenSpot-Pro auto-downloads on first eval call
# (Or pre-warm: huggingface-cli download <repo> --local-dir ~/screenspot_pro)
```

Expect ~3-4GB ScreenSpot-Pro, ~few GB OS-Atlas linux subset, ~8GB model.

---

## Lane A: Sampling

**Smoke test first — DO NOT skip.** If this fails you waste H100-hours.

```bash
# Smoke: 100 items, 1 shard
python sample/sample_northstar.py --n 100 --shard 0 --total-shards 1

# Verify output exists + format sane
ls samples/shard_0/sft.jsonl
head -1 samples/shard_0/sft.jsonl | python -m json.tool
```

If smoke passes, scale up:

```bash
# H100x4: launch 4 shards, one per GPU
./sample/launch_shards.sh 4

# Monitor
python sample/status.py            # shows progress, hit rate, ETA
tail -f samples/shard_0/log.txt

# Kill a shard if you need its GPU back (e.g. for training)
./sample/kill_shard.sh 0
```

Output lands in `samples/shard_<i>/sft.jsonl` — one line per accepted example. `done_ids.txt` tracks resume state per shard.

---

## Lane B: Train + Eval

Trigger training **manually** when status dashboard shows ~5-10k accepted examples.

```bash
# 1. Snapshot current SFT data (atomic — won't race with sampler writes)
python train/freeze_snapshot.py --run-id run_001

# 2. Train (uses snapshot from step 1)
./train_run.sh run_001
# - merges shard_*/sft.jsonl into one snapshot
# - LoRA SFT: r=32, alpha=64, lr=1e-4 cosine, bs=4 grad-accum=8, 1 epoch
# - vision tower frozen
# - writes adapter to adapters/run_001/

# 3. Subset eval (200 stratified items, ~5min on H100)
python eval/eval_screenspot_pro.py --adapter adapters/run_001 --subset 200

# 4. Update leaderboard
python eval/leaderboard.py

# 5. If best so far, run full eval
python eval/finalize.py --adapter adapters/run_001  # full 1581 items, ~30min
```

**GPU stealing pattern (single-instance, if you only have 1 free GPU):**
```bash
./sample/kill_shard.sh 0      # free GPU 0
./train_run.sh run_002        # uses GPU 0
./sample/launch_shards.sh 4   # restart all shards once train done
```

---

## What's locked — DO NOT change

| Thing | Value | Why |
|-------|-------|-----|
| Action format | `<tool_call>{"name":"computer_use","arguments":{"type":"click","x":<int>,"y":<int>}}</tool_call>` | Model RL-trained for this exact schema |
| Coords | 0-999 normalized, divide by 1000 → pixel | Northstar convention |
| Tool spec | Pass via vLLM `LLM.chat(tools=...)` | Required, model expects it |
| Image preproc | cap 2M pixels via `mm_processor_kwargs.max_pixels` | Memory + matches model train |
| vLLM | `enable_prefix_caching=True`, `dtype=bfloat16`, `max_num_seqs=16` | Tested |
| LoRA | r=32, alpha=64, target attn+MLP of LM only | Vision frozen |
| Train | bs=4, grad-accum=8, lr=1e-4 cosine, warmup 5%, 1 epoch | Tested |
| Loss mask | response tokens only (standard trl SFT) | |
| Subset eval | 200 stratified | Fast iteration |
| Final eval | full 1581 | Real number |

`northstar_format.py` is single source of truth for tool spec + action parser. Both sample and eval import it. **If you touch it, run `pytest tests/` before anything else.**

---

## My state (cd-b200, what I'm doing right now)

- Repo cloned at `~/click-distill/`
- All deps installed, vLLM 0.20.1 + torch 2.11+cu130, sm_100 (Blackwell) verified
- Northstar model + ScreenSpot-Pro + OS-Atlas linux subset all downloaded
- About to run smoke test, then scale to 50k pool with 1 shard (B200 = 1 GPU)
- Est sampling rate: 30-40 completions/sec (unmeasured, will know after smoke)

---

## Brev instance status

| Instance | Type | State | Notes |
|----------|------|-------|-------|
| cd-b200 | verda B200 1× 192GB | ALIVE | Me. Running. |
| cd-h100x4 | hyperstack H100x4 | FAILURE (billing) | Don't kill yet — gabriel checking |
| cd-h100 | dmz h100x2 pcie | STARTING | Might come up — could be your fallback |
| (older) | hyperstack 2x/4x | DELETED | Capacity issues |

**Budget**: ~$6-10 spent of $40 so far.

---

## Risks (open)

| Risk | Mitigation |
|------|-----------|
| Sampling rate too slow on B200 | Measure after smoke; if <20/s investigate `max_num_seqs` |
| peft/trl hiccups on Blackwell | Fall back to plain HF Trainer (no trl) — code path exists |
| Time pressure | If smoke >5min, skip scale-up, train on whatever's accumulated |
| You + me overwrite each other's adapters | Use distinct `--run-id` (e.g. `run_h100_001`, `run_b200_001`) |

## Risks (already mitigated)

- vLLM Blackwell support: verified working
- Sample/eval action format mismatch: shared `northstar_format.py`
- Atomic writes / crash resume: per-item fsync + `done_ids.txt`
- 0-hit items waste compute: skipped permanently after first attempt

---

## If you want to merge data with mine at the end

```bash
# On your H100, after some sampling
rsync -av samples/ <gabriel-b200-host>:~/click-distill/merged_samples/h100/
# Or: aws s3 sync samples/ s3://<bucket>/click-distill/h100/
```

Then on whichever instance trains the final adapter:
```bash
cat samples/shard_*/sft.jsonl merged_samples/h100/shard_*/sft.jsonl > combined.jsonl
# Point train_run.sh at combined.jsonl (edit `--data` arg)
```

Don't bother unless we have >30min slack. Independent runs is the safer play.

---

## Diversification idea (if you have time)

To diversify pool vs my run, you could:
- Higher N samples per item (e.g. 16 instead of 8) → more rejection candidates
- Wider temp range (e.g. 0.7-1.2 instead of 0.7-1.0)
- Different OS-Atlas subset (`--n 50000 --offset 50000`)

Pass via flags to `sample/sample_northstar.py` — see `--help`. Don't fork the code.

---

## Decision log (for context, not action)

1. RSSD (rejection-sampled self-distill), not DPO/RLHF — simpler, time-bound
2. Hash-mod sharding — no coordinator
3. Per-item synchronous flush — crash-safe
4. 50k input pool default — covers diversity, fits in time
5. Manual training trigger — human-in-loop quality gate
6. Skip 0-hit items permanently — don't waste compute
7. Versioned snapshots/adapters/leaderboard — no overwrite ambiguity
8. GPU stealing on single instance — no scheduler complexity
9. 200-item stratified subset for iter, full 1581 for final — speed vs truth
10. Split status.py + leaderboard.py — single responsibility

---

## Sanity checklist before claiming "done"

- [ ] `pytest tests/` green
- [ ] Smoke test on 100 items completed end-to-end
- [ ] At least one adapter trained
- [ ] Subset eval (200 items) shows >baseline
- [ ] Full eval (1581) on best adapter
- [ ] Leaderboard committed to repo
- [ ] Final adapter pushed somewhere accessible (HF hub? S3?)

---

**Ping gabriel on Slack/Discord if anything blocks.** Good luck.
