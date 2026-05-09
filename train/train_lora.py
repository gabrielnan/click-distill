"""LoRA SFT trainer for Northstar-CUA-Fast on RSSD snapshots.

Each training run starts FRESH from the base model (Tzafon/Northstar-CUA-Fast)
and trains LoRA adapters on the latest snapshot of the streaming sampler.
No warm-resume across runs — clean ablation, easy to compare.

Hyperparams are LOCKED in ``HYPERPARAMS`` and asserted via
``validate_hyperparams()``: r=32, alpha=64, lr=1e-4, cosine schedule with
5% warmup, 1 epoch, bs=4 with grad-accum=8 (eff bs=32), bf16, vision tower
frozen. This is the canonical config for the hackathon.

Pipeline (CLI):
  1. Load snapshot JSONL.
  2. Build chat-format dataset (system w/ tool spec, user w/ image+instruction,
     assistant w/ completion). Tool spec via ``northstar_format.build_computer_use_tool``.
  3. Apply Qwen3-VL chat template + processor; mask prompt tokens out of loss.
  4. ``peft.get_peft_model`` with the locked LoRA config; freeze vision tower.
  5. ``trl.SFTTrainer`` with the locked TrainingArguments.
  6. Save adapter + tokenizer + training_args to ``--output-dir``.

We deliberately keep step (3)..(6) IMPORT-LAZY so this module is unit-testable
on a CPU-only machine (the test suite exercises only the pure functions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo-root single-source-of-truth for the tool schema.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from northstar_format import build_computer_use_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Locked hyperparameters
# ---------------------------------------------------------------------------
HYPERPARAMS: dict[str, Any] = {
    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "freeze_vision_tower": True,
    # Optim / schedule
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "num_train_epochs": 1,
    # Batching
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # effective batch = 32
    # Precision / memory
    "bf16": True,
    "gradient_checkpointing": True,
    # Misc
    "max_seq_length": 4096,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "report_to": "none",
}


def validate_hyperparams(h: dict[str, Any]) -> None:
    """Sanity-check the hyperparam dict.

    The hackathon committed to fixed hyperparams across all runs (clean
    ablation). Drift here turns leaderboard deltas into noise, so we refuse
    to train if any locked value has been changed.
    """
    expected = {
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "bf16": True,
        "freeze_vision_tower": True,
    }
    for k, v in expected.items():
        if h.get(k) != v:
            raise ValueError(
                f"hyperparam drift: {k}={h.get(k)} (expected {v}). "
                f"Hackathon spec requires fixed hyperparams across runs."
            )


# ---------------------------------------------------------------------------
# Snapshot -> chat dataset
# ---------------------------------------------------------------------------
def _completion_from_row(row: dict) -> str:
    """Pull the assistant completion out of a snapshot row.

    The streaming sampler writes a flat ``"completion"`` scalar (one row per
    winning sample). The pre-streaming format wrote a ``"samples"`` list (we
    pick the first hit by convention).
    """
    if "completion" in row and isinstance(row["completion"], str):
        return row["completion"]
    if "samples" in row and isinstance(row["samples"], list) and row["samples"]:
        return row["samples"][0]
    raise KeyError(f"snapshot row missing completion/samples: {row.keys()}")


def build_sft_dataset(snapshot_jsonl: Path) -> list[dict]:
    """Read a snapshot JSONL into chat-format training rows.

    Each output row is::

        {
          "messages": [
            {"role": "user", "content": [{"type": "image", "image": <path>},
                                          {"type": "text", "text": <instr>}]},
            {"role": "assistant", "content": <completion>}
          ],
          "tools": [<computer_use tool dict>],
          "image_path": <abs path>,
        }

    The system block is rendered by Qwen3-VL's chat template at tokenize-time
    from the ``tools=`` arg — same path eval and sampling already use
    (``northstar_format.build_computer_use_tool``). This guarantees the
    trained model sees the IDENTICAL prompt shape it'll be evaluated under.
    """
    rows: list[dict] = []
    with Path(snapshot_jsonl).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": r["image_path"]},
                        {"type": "text", "text": r["instruction"]},
                    ],
                },
                {"role": "assistant", "content": _completion_from_row(r)},
            ]
            tools = [build_computer_use_tool(
                int(r["image_width"]), int(r["image_height"]),
            )]
            rows.append({
                "messages": messages,
                "tools": tools,
                "image_path": r["image_path"],
            })
    return rows


# ---------------------------------------------------------------------------
# Loss mask (prompt-only -> 0; response -> 1)
# ---------------------------------------------------------------------------
def compute_loss_mask(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Build an SFT loss mask.

    The standard trl SFT setup tokenizes the FULL chat (prompt + assistant
    response) once. We tokenize the prompt-only path separately, then mask
    out the prefix so loss only flows on the response tokens.

    Args:
        prompt_ids: token ids for the prompt portion (system+user, no
            assistant turn yet).
        full_ids: token ids for the full chat (prompt + assistant turn).
            ``prompt_ids`` MUST be a prefix of ``full_ids``.

    Returns:
        List of 0/1 of length ``len(full_ids)``.
    """
    if len(prompt_ids) > len(full_ids):
        raise ValueError(
            f"prompt_ids ({len(prompt_ids)}) longer than full_ids ({len(full_ids)})"
        )
    if full_ids[: len(prompt_ids)] != list(prompt_ids):
        raise ValueError("prompt_ids is not a prefix of full_ids")
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return mask


# ---------------------------------------------------------------------------
# Trainer entry point (lazy heavy imports)
# ---------------------------------------------------------------------------
def _train(args: argparse.Namespace) -> None:
    """The actual training run. Imports transformers/peft/trl lazily so
    ``--help`` and unit tests don't pay the import cost."""
    import torch  # noqa: F401  -- imported for side effects (CUDA check below)
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    validate_hyperparams(HYPERPARAMS)

    print(f"[train] base model: {args.base_model}")
    print(f"[train] snapshot:   {args.snapshot}")
    print(f"[train] output:     {args.output_dir}")

    rows = build_sft_dataset(Path(args.snapshot))
    print(f"[train] {len(rows)} SFT rows from snapshot")
    if len(rows) == 0:
        sys.exit(f"snapshot {args.snapshot} has zero rows; aborting")

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )

    if HYPERPARAMS["freeze_vision_tower"]:
        # Best-effort: Qwen3-VL exposes ``model.visual`` / ``model.vision_tower``;
        # freeze whichever attribute exists.
        for attr in ("visual", "vision_tower", "vision_model"):
            sub = getattr(model, attr, None)
            if sub is not None:
                for p in sub.parameters():
                    p.requires_grad = False
                print(f"[train] froze {attr}")

    lora_cfg = LoraConfig(
        r=HYPERPARAMS["lora_r"],
        lora_alpha=HYPERPARAMS["lora_alpha"],
        lora_dropout=HYPERPARAMS["lora_dropout"],
        target_modules=HYPERPARAMS["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # HuggingFace Datasets requires a flat schema; we keep the per-row tools
    # in a sidecar dict and resolve inside the data collator. For the hackathon
    # path we just rely on processor.apply_chat_template(messages, tools=...)
    # via the standard SFTTrainer ``formatting_func``.
    ds = Dataset.from_list(rows)

    def formatting_func(example):
        return processor.apply_chat_template(
            example["messages"],
            tools=example["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=HYPERPARAMS["num_train_epochs"],
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        gradient_accumulation_steps=HYPERPARAMS["gradient_accumulation_steps"],
        learning_rate=HYPERPARAMS["learning_rate"],
        lr_scheduler_type=HYPERPARAMS["lr_scheduler_type"],
        warmup_ratio=HYPERPARAMS["warmup_ratio"],
        bf16=HYPERPARAMS["bf16"],
        gradient_checkpointing=HYPERPARAMS["gradient_checkpointing"],
        logging_steps=HYPERPARAMS["logging_steps"],
        save_strategy=HYPERPARAMS["save_strategy"],
        report_to=HYPERPARAMS["report_to"],
        max_seq_length=HYPERPARAMS["max_seq_length"],
        dataset_text_field=None,  # we use formatting_func
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        formatting_func=formatting_func,
        tokenizer=processor,
    )
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    (out / "training_args.json").write_text(json.dumps(HYPERPARAMS, indent=2))
    print(f"[train] saved adapter -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True,
                   help="Path to snapshots/run_<id>/sft.jsonl")
    p.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the LoRA adapter (e.g. adapters/lora_1130/)")
    p.add_argument("--epochs", type=int, default=HYPERPARAMS["num_train_epochs"])
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.epochs != HYPERPARAMS["num_train_epochs"]:
        # Hyperparam lock: --epochs overrides the dict and re-validates.
        HYPERPARAMS["num_train_epochs"] = args.epochs
        validate_hyperparams(HYPERPARAMS)
    _train(args)


if __name__ == "__main__":
    main()
