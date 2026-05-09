"""LoRA training script test (Component 5).

We test the PURE FUNCTIONS that don't require a GPU:

* ``build_sft_dataset(snapshot_jsonl)`` — formats each row as a chat with
  the canonical Northstar tool spec (system w/ tool spec via the
  apply_chat_template path, user w/ image+instruction, assistant w/
  completion).
* ``compute_loss_mask(prompt_ids, full_ids)`` — produces a mask where the
  prompt tokens are 0 and the response tokens are 1 (standard SFT loss).
* ``HYPERPARAMS`` — the locked-in config (r=32, alpha=64, lr=1e-4 etc.)
  validates against expected values.

The actual ``transformers``/``trl``/``peft`` integration is mocked: we don't
load Qwen3-VL on this Mac.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.train_lora import (
    HYPERPARAMS,
    build_sft_dataset,
    compute_loss_mask,
    validate_hyperparams,
)


def _write_snapshot(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "sft.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


# ---------------------------------------------------------------------------
# Hyperparam config
# ---------------------------------------------------------------------------
def test_hyperparams_locked_values():
    """Hyperparams must match PLAN.md / spec for clean ablation."""
    assert HYPERPARAMS["lora_r"] == 32
    assert HYPERPARAMS["lora_alpha"] == 64
    assert HYPERPARAMS["learning_rate"] == 1e-4
    assert HYPERPARAMS["lr_scheduler_type"] == "cosine"
    assert HYPERPARAMS["warmup_ratio"] == 0.05
    assert HYPERPARAMS["num_train_epochs"] == 1
    assert HYPERPARAMS["per_device_train_batch_size"] == 4
    assert HYPERPARAMS["gradient_accumulation_steps"] == 8
    assert HYPERPARAMS["bf16"] is True
    assert HYPERPARAMS["freeze_vision_tower"] is True


def test_validate_hyperparams_passes_for_canonical():
    validate_hyperparams(HYPERPARAMS)  # must not raise


def test_validate_hyperparams_rejects_drift():
    bad = dict(HYPERPARAMS, learning_rate=5e-3)
    with pytest.raises(ValueError):
        validate_hyperparams(bad)


# ---------------------------------------------------------------------------
# build_sft_dataset
# ---------------------------------------------------------------------------
def test_build_sft_dataset_one_row(tmp_path: Path):
    snap = _write_snapshot(tmp_path, [{
        "item_id": "1",
        "instruction": "click search bar",
        "image_path": "/img/a.png",
        "image_width": 1024,
        "image_height": 768,
        "completion": (
            '<tool_call>{"name":"computer_use","arguments":'
            '{"type":"click","x":234,"y":567}}</tool_call>'
        ),
    }])
    rows = build_sft_dataset(snap)
    assert len(rows) == 1
    row = rows[0]

    # Each row carries (messages, tools, image_path) for the trainer to consume.
    assert "messages" in row
    assert "tools" in row
    assert row["image_path"] == "/img/a.png"

    # User message: image + text in correct order.
    user_msg = next(m for m in row["messages"] if m["role"] == "user")
    parts = user_msg["content"]
    assert parts[0]["type"] == "image"
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == "click search bar"

    # Assistant message: the full completion.
    assistant_msg = next(m for m in row["messages"] if m["role"] == "assistant")
    assert "computer_use" in assistant_msg["content"]
    assert "234" in assistant_msg["content"]

    # Tool spec: built via the canonical northstar_format helper, with the
    # image's width/height surfaced.
    assert len(row["tools"]) == 1
    tool = row["tools"][0]
    assert tool["function"]["name"] == "computer_use"
    assert tool["function"]["display_width"] == 1024
    assert tool["function"]["display_height"] == 768
    # Flat x/y, never coordinate-array.
    props = tool["function"]["parameters"]["properties"]
    assert "x" in props and "y" in props
    assert "coordinate" not in props


def test_build_sft_dataset_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "sft.jsonl"
    p.write_text(
        json.dumps({
            "item_id": "1",
            "instruction": "x", "image_path": "/a.png",
            "image_width": 1024, "image_height": 768,
            "completion": "<tool_call>...</tool_call>",
        }) + "\n\n  \n" + json.dumps({
            "item_id": "2",
            "instruction": "y", "image_path": "/b.png",
            "image_width": 1024, "image_height": 768,
            "completion": "<tool_call>...</tool_call>",
        }) + "\n",
    )
    rows = build_sft_dataset(p)
    assert len(rows) == 2


def test_build_sft_dataset_uses_legacy_completion_field_aliases(tmp_path: Path):
    """If the snapshot uses 'samples' (Lane A's pre-streaming format), accept
    one sample per row. The streaming sampler writes a 'completion' scalar."""
    p = tmp_path / "sft.jsonl"
    rec = {
        "item_id": "x",
        "instruction": "click",
        "image_path": "/x.png",
        "image_width": 1024,
        "image_height": 768,
        # Old format: a list of samples; the WINNING one is the canonical row.
        "samples": ['<tool_call>{"name":"computer_use","arguments":{"type":"click","x":1,"y":2}}</tool_call>'],
    }
    with p.open("w") as fh:
        fh.write(json.dumps(rec) + "\n")
    rows = build_sft_dataset(p)
    assert len(rows) == 1
    asst = next(m for m in rows[0]["messages"] if m["role"] == "assistant")
    assert "computer_use" in asst["content"]


# ---------------------------------------------------------------------------
# compute_loss_mask
# ---------------------------------------------------------------------------
def test_compute_loss_mask_simple():
    """Prompt tokens get loss=0; response tokens get loss=1."""
    prompt_ids = [10, 20, 30]              # length 3
    full_ids = [10, 20, 30, 40, 50, 60]    # length 6
    mask = compute_loss_mask(prompt_ids, full_ids)
    assert mask == [0, 0, 0, 1, 1, 1]


def test_compute_loss_mask_empty_response():
    """If full_ids == prompt_ids, every token is masked out."""
    mask = compute_loss_mask([1, 2, 3], [1, 2, 3])
    assert mask == [0, 0, 0]


def test_compute_loss_mask_length_matches_full():
    prompt_ids = list(range(50))
    full_ids = list(range(80))
    mask = compute_loss_mask(prompt_ids, full_ids)
    assert len(mask) == 80
    assert sum(mask[:50]) == 0
    assert sum(mask[50:]) == 30


def test_compute_loss_mask_rejects_inconsistent_prefix():
    """If prompt_ids isn't a prefix of full_ids, that's a bug — refuse silently."""
    prompt_ids = [1, 2, 3, 99]   # 99 conflicts with full[3]=4
    full_ids = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        compute_loss_mask(prompt_ids, full_ids)
