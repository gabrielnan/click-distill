"""Shared pytest fixtures + path setup for the click-distill test suite.

These tests run **without** vLLM, GPUs, or network access. We mock anything
heavyweight (vLLM ``LLM``, ``os.fsync``, ``nvidia-smi``) and exercise only the
streaming-pipeline glue: shard filtering, per-item flushing, resume, status
rendering, and CLI launchers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "sample"
DATA_DIR = REPO_ROOT / "data"

# Make the repo root importable so tests can do ``from sample import ...``,
# ``from northstar_format import ...``, etc.
for p in (REPO_ROOT, SAMPLE_DIR, DATA_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@pytest.fixture()
def tmp_jsonl_factory(tmp_path: Path):
    """Returns a callable that writes a list-of-dicts as a JSONL and returns the path."""

    def _make(name: str, rows: list[dict]) -> Path:
        out = tmp_path / name
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return out

    return _make


def _make_item(idx: int, *, hits: int = 0, samples: int = 8) -> dict:
    """Synthetic Northstar input item with deterministic ground-truth bbox.

    The bbox is set up so that completions hitting (50, 50) are inside, and
    (900, 900) are outside. Useful for tests that need to control hit/miss
    independently of the real OS-Atlas data.
    """
    return {
        "item_id": str(idx),
        "instruction": f"click element {idx}",
        "image_path": f"/fake/img_{idx}.png",
        "bbox": [40, 40, 60, 60],  # pixel-space xyxy
        "image_width": 1000,
        "image_height": 1000,
    }


@pytest.fixture()
def make_item():
    return _make_item


def make_completion_hit() -> str:
    """A canonical-format completion that lands inside ``[40,40,60,60]``."""
    return (
        '<tool_call>{"name":"computer_use","arguments":'
        '{"type":"click","x":50,"y":50}}</tool_call>'
    )


def make_completion_miss() -> str:
    """A canonical-format completion that lands outside ``[40,40,60,60]``."""
    return (
        '<tool_call>{"name":"computer_use","arguments":'
        '{"type":"click","x":900,"y":900}}</tool_call>'
    )


@pytest.fixture()
def hit_completion():
    return make_completion_hit


@pytest.fixture()
def miss_completion():
    return make_completion_miss
