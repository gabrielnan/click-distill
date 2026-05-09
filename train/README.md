# train/ — LoRA SFT + snapshot freezer

This directory owns the training half of the streaming RSSD pipeline.

```
train/
├── freeze_snapshot.py    # samples/shard_*/sft.jsonl  ->  snapshots/run_<id>/
├── train_lora.py         # LoRA SFT on a snapshot, writes adapters/lora_<id>/
└── README.md             # this file
```

## Streaming-pipeline contract

A separate sampler daemon (in `sample/`) writes
`samples/shard_<id>/sft.jsonl` lines continuously. We do NOT consume that
directory directly — the file would change underneath the trainer. Instead:

1. `freeze_snapshot.py` reads every shard's `sft.jsonl`, concatenates the
   union into `snapshots/run_<id>/sft.jsonl`, and writes a tiny
   `snapshot_meta.json` with provenance (`{run_id, created_at,
   source_shards, total_sft_items}`).
2. `train_lora.py` ingests that frozen snapshot — never the live `samples/`
   dir.

Sampling continues writing to `samples/` while training runs.

## Training run = fresh from base + LoRA on snapshot

Each run starts FRESH from `Tzafon/Northstar-CUA-Fast`. No warm-resume
across runs — the only thing that grows between runs is the snapshot size,
which makes leaderboard deltas easy to interpret ("how does N more SFT
samples buy us?").

## Locked hyperparameters

Defined in `train_lora.HYPERPARAMS` and asserted by `validate_hyperparams()`.
Drift will refuse to train (so leaderboard deltas always mean the same thing):

| param                          | value                              |
|--------------------------------|------------------------------------|
| LoRA r                         | 32                                 |
| LoRA alpha                     | 64                                 |
| LoRA dropout                   | 0.05                               |
| target modules                 | q/k/v/o/gate/up/down_proj          |
| learning rate                  | 1e-4 (cosine, 5% warmup)           |
| epochs                         | 1                                  |
| eff batch size                 | 32 (bs=4 × grad-accum=8)           |
| precision                      | bf16                               |
| gradient checkpointing         | on                                 |
| vision tower                   | frozen                             |
| max seq length                 | 4096                               |

## Loss masking

Standard `trl.SFTTrainer` setup: tokenize the full chat (system + user +
assistant), mask everything except the assistant turn. The prompt block
includes the `computer_use` tool spec rendered by the Qwen3-VL chat template
(via the `tools=` arg), so the model sees the IDENTICAL shape it will be
evaluated under.

`compute_loss_mask(prompt_ids, full_ids)` is exposed as a helper and unit
tested in `tests/test_train_lora.py`.

## CLI

```bash
python train/freeze_snapshot.py --run-id 1130
# -> snapshots/run_1130/sft.jsonl + snapshot_meta.json

python train/train_lora.py \
    --snapshot snapshots/run_1130/sft.jsonl \
    --base-model Tzafon/Northstar-CUA-Fast \
    --output-dir adapters/lora_1130
# -> adapters/lora_1130/{adapter_model.safetensors, training_args.json, ...}
```

In practice these are wrapped by `train_run.sh <run_id>` at the repo root,
which also kills sampler shard 0 to free GPU 0, runs the subset eval, and
appends the leaderboard.

## Why a flat snapshot file

`freeze_snapshot.py` collapses N per-shard files into ONE
`sft.jsonl` so the trainer ingests it as a single dataset. The alternative
(globbing `samples/shard_*/sft.jsonl` directly inside the trainer) couples
training to the shard layout and to whatever the sampler is doing right now
— both of which we want decoupled for reproducibility.
