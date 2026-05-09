"""Tests for sample/launch_shards.sh + sample/kill_shard.sh (Component 4).

We don't actually exec vLLM — the shell scripts call ``--shard-cmd`` (or our
``WORKER_CMD`` env override) so the tests can plug in a stub that just records
``CUDA_VISIBLE_DEVICES`` + args + writes a long-running pid file.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SH = REPO_ROOT / "sample" / "launch_shards.sh"
KILL_SH = REPO_ROOT / "sample" / "kill_shard.sh"


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _kill_pid(pid: int) -> None:
    """Best-effort kill — used in test teardown."""
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@pytest.fixture()
def stub_worker(tmp_path: Path) -> Path:
    """A tiny worker that simulates a shard: writes pid + env to its shard dir
    and sleeps until killed."""
    stub = tmp_path / "stub_worker.sh"
    stub.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        '# usage: stub_worker.sh <shard_dir> <shard> <num_shards>\n'
        'shard_dir="$1"\n'
        'mkdir -p "$shard_dir"\n'
        'echo $$ > "$shard_dir/pid"\n'
        'echo "shard=$2 num_shards=$3 cuda=${CUDA_VISIBLE_DEVICES:-unset}" \\\n'
        '    > "$shard_dir/env.txt"\n'
        'while true; do sleep 1; done\n'
    )
    stub.chmod(0o755)
    return stub


def test_launch_shards_creates_pids_and_sets_cuda(tmp_path: Path, stub_worker: Path):
    """launch_shards.sh --num-shards 4 --gpus 0,1,2,3 must:
      - create samples/shard_<i>/pid for i in 0..3
      - set CUDA_VISIBLE_DEVICES correctly per shard
      - write logs to samples/shard_<i>/log.txt
    """
    samples_root = tmp_path / "samples"
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text("")  # empty — stub doesn't read it

    env = os.environ.copy()
    env["WORKER_CMD"] = str(stub_worker)

    proc = subprocess.run(
        [
            "bash", str(LAUNCH_SH),
            "--num-shards", "4",
            "--gpus", "0,1,2,3",
            "--samples-root", str(samples_root),
            "--input", str(input_jsonl),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, f"launch failed: {proc.stderr}"

    pids: list[int] = []
    try:
        # All 4 pid files should appear within a few seconds.
        for shard in range(4):
            pid_file = samples_root / f"shard_{shard}" / "pid"
            assert _wait_for(lambda p=pid_file: p.exists()), f"missing {pid_file}"
            pid = int(pid_file.read_text().strip())
            pids.append(pid)

            env_file = samples_root / f"shard_{shard}" / "env.txt"
            assert _wait_for(lambda f=env_file: f.exists()), f"missing {env_file}"
            content = env_file.read_text()
            assert f"shard={shard}" in content
            assert "num_shards=4" in content
            assert f"cuda={shard}" in content, f"shard {shard} got: {content}"
    finally:
        for pid in pids:
            _kill_pid(pid)


def test_kill_shard_terminates_only_target(tmp_path: Path, stub_worker: Path):
    """kill_shard.sh 0 must kill shard 0's pid only — others must keep running."""
    samples_root = tmp_path / "samples"
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text("")

    env = os.environ.copy()
    env["WORKER_CMD"] = str(stub_worker)

    proc = subprocess.run(
        [
            "bash", str(LAUNCH_SH),
            "--num-shards", "2",
            "--gpus", "0,1",
            "--samples-root", str(samples_root),
            "--input", str(input_jsonl),
        ],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr

    pid0_file = samples_root / "shard_0" / "pid"
    pid1_file = samples_root / "shard_1" / "pid"
    assert _wait_for(lambda: pid0_file.exists() and pid1_file.exists())
    pid0 = int(pid0_file.read_text().strip())
    pid1 = int(pid1_file.read_text().strip())

    try:
        # Kill shard 0.
        kill_proc = subprocess.run(
            ["bash", str(KILL_SH), "--samples-root", str(samples_root), "0"],
            capture_output=True, text=True, timeout=5,
        )
        assert kill_proc.returncode == 0, kill_proc.stderr

        # Shard 0 should die; shard 1 should still be running.
        def _alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                return False

        assert _wait_for(lambda: not _alive(pid0)), "shard 0 still alive"
        assert _alive(pid1), "shard 1 was killed by mistake"
    finally:
        _kill_pid(pid0)
        _kill_pid(pid1)


def test_launch_shards_validates_gpu_count(tmp_path: Path, stub_worker: Path):
    """``--num-shards`` and the comma-list passed to ``--gpus`` must agree."""
    samples_root = tmp_path / "samples"
    env = os.environ.copy()
    env["WORKER_CMD"] = str(stub_worker)

    proc = subprocess.run(
        [
            "bash", str(LAUNCH_SH),
            "--num-shards", "4",
            "--gpus", "0,1",  # only 2 — mismatch
            "--samples-root", str(samples_root),
            "--input", str(tmp_path / "x.jsonl"),
        ],
        env=env, capture_output=True, text=True, timeout=5,
    )
    assert proc.returncode != 0
    assert "gpus" in (proc.stderr + proc.stdout).lower()
