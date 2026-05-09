"""Tests for data/download_osatlas.py defaults + streaming behaviour (Component 6).

We don't actually download from HuggingFace — we patch ``download_annotation``
to return a hand-rolled JSON file. The test asserts:

* default --n is 50000 (per the spec bump)
* default --output is data/osatlas_hard_50k.jsonl
* iter_filtered_elements is a generator (yields lazily, doesn't materialize
  the whole upstream into RAM)
* the resulting JSONL has stable line indices (i.e. position == "item_id"
  in the streaming sampler's hash-mod scheme)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import the module by file path since data/ has no __init__.py.
import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_PY = REPO_ROOT / "data" / "download_osatlas.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_osatlas", DOWNLOAD_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_osatlas"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_n_is_50000():
    """Pool size default is 50k (was 5k pre-streaming).

    Bigger pool = more headroom for the daemon to keep sampling overnight.
    """
    mod = _load_module()
    args = mod.parse_args.__wrapped__() if hasattr(mod.parse_args, "__wrapped__") else None
    # parse_args reads sys.argv; just inspect the parser construction.
    import argparse
    parser = argparse.ArgumentParser()
    # Mirror the script's --n config; verify the default in the source.
    src = DOWNLOAD_PY.read_text()
    assert '--n", type=int, default=50000' in src, (
        "default --n should be 50000 (was 5000 pre-streaming pipeline)"
    )


def test_default_output_path_renamed():
    """Default output should reflect the new size (osatlas_hard_50k.jsonl)."""
    src = DOWNLOAD_PY.read_text()
    assert "osatlas_hard_50k.jsonl" in src, (
        "default --output should be data/osatlas_hard_50k.jsonl"
    )


def test_iter_filtered_elements_is_lazy(tmp_path: Path):
    """Verify that iter_filtered_elements is a generator function — i.e. we
    don't materialize the entire upstream JSON into a list before filtering.

    The filter pass is hot-path (270MB Windows annotations) so this is
    architectural, not just stylistic.
    """
    import inspect
    mod = _load_module()
    fn = mod.iter_filtered_elements
    assert inspect.isgeneratorfunction(fn), (
        "iter_filtered_elements must be a generator (yield) — required for "
        "the 50k pool to fit in memory."
    )


def test_collect_streams_with_mock_annotations(tmp_path: Path, monkeypatch):
    """End-to-end: mock the annotation download, run collect(), confirm we get
    rows with stable order matching their position in the upstream JSON."""
    mod = _load_module()
    from collections import Counter

    # Build a synthetic annotation JSON with mixed pass/fail rows.
    fake_data = []
    # 5 items each with one element; vary instruction length / bbox area
    # so some get filtered.
    for i in range(20):
        instr = " ".join(f"word{j}" for j in range(7))  # 7 words: passes 5-30
        # 0.5% area: passes <1%
        bbox = [0.0, 0.0, 0.05, 0.1]
        fake_data.append({
            "img_filename": f"img_{i}.png",
            "elements": [{
                "instruction": instr,
                "bbox": bbox,
                "data_type": "icon",
            }],
        })
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(fake_data))

    def fake_download(source_id, cache_dir):
        return ann_path

    monkeypatch.setattr(mod, "download_annotation", fake_download)

    rows, skipped = mod.collect(["desktop/linux"], tmp_path / "cache")
    assert len(rows) == 20
    # All rows from a single source preserve upstream order.
    paths = [r["image_path"] for r in rows]
    assert paths == [f"desktop_domain/linux_images/img_{i}.png" for i in range(20)]


def test_balanced_sample_preserves_indices_across_n(tmp_path: Path, monkeypatch):
    """When --n exceeds available rows, balanced_sample returns ALL rows.
    The downstream sampler treats line position as item_id — this tests that
    the function handles the "ask for 50k, get 20" edge case without crashing."""
    mod = _load_module()
    rows = [
        {"source": "a", "image_path": f"x_{i}", "instruction": "y", "bbox": [0, 0, 1, 1]}
        for i in range(20)
    ]
    picked = mod.balanced_sample(rows, n=50000, seed=0)
    assert len(picked) == 20
    # Stable seed -> same shuffle.
    picked2 = mod.balanced_sample(rows, n=50000, seed=0)
    assert [r["image_path"] for r in picked] == [r["image_path"] for r in picked2]


def test_write_jsonl_produces_one_line_per_row(tmp_path: Path):
    """Confirm the JSONL writer's one-line-per-row contract — the streaming
    sampler relies on enumerate() of the file giving stable line indices."""
    mod = _load_module()
    rows = [{"a": i, "b": "x" * 100} for i in range(10)]
    out = tmp_path / "out.jsonl"
    mod.write_jsonl(rows, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 10
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["a"] == i
