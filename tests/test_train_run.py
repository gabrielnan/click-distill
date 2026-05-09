"""Orchestrator test (Component 6).

``train_run.sh <run_id>`` glues together: kill sampler shard 0, freeze
snapshot, train (mocked), subset-eval (mocked), append leaderboard, restart
shard 0. We exercise it end-to-end with stubs for the heavy parts.

The other agent owns ``sample/launch_shards.sh`` and ``sample/kill_shard.sh``.
We stub those at ``$REPO/.test_stubs/sample/{launch_shards,kill_shard}.sh``
and prepend that dir to ``PATH`` (well: we set the ``SAMPLE_DIR`` env var
that ``train_run.sh`` honors).

Likewise we stub ``train/train_lora.py`` and ``eval/eval_screenspot_pro.py``
via env-var hooks (``TRAIN_LORA_CMD`` / ``EVAL_CMD``) that the orchestrator
uses if set, falling back to the real scripts otherwise. This is the same
pattern Lane A uses for its launch_shards stub.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_RUN_SH = REPO_ROOT / "train_run.sh"


@pytest.fixture
def stage(tmp_path: Path):
    """Set up an isolated test world with stubs."""
    # Working dirs the orchestrator will read/write.
    samples = tmp_path / "samples"
    snapshots = tmp_path / "snapshots"
    adapters = tmp_path / "adapters"
    results = tmp_path / "results"
    bin_dir = tmp_path / "bin"

    for d in (samples / "shard_0", samples / "shard_1", snapshots, adapters, results, bin_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Pre-seed shard sft.jsonls.
    (samples / "shard_0" / "sft.jsonl").write_text(
        json.dumps({"item_id": "0_a", "instruction": "x", "image_path": "/a.png",
                    "image_width": 1024, "image_height": 768,
                    "completion": "<tool_call>...</tool_call>"}) + "\n"
    )
    (samples / "shard_0" / "status.json").write_text(json.dumps({"shard_id": 0}))
    (samples / "shard_1" / "sft.jsonl").write_text(
        json.dumps({"item_id": "1_a", "instruction": "y", "image_path": "/b.png",
                    "image_width": 1024, "image_height": 768,
                    "completion": "<tool_call>...</tool_call>"}) + "\n"
    )
    (samples / "shard_1" / "status.json").write_text(json.dumps({"shard_id": 1}))

    # Sampler shard 0 stub: a long-sleep child that records its PID.
    pidfile = samples / "shard_0" / "shard.pid"
    sampler = bin_dir / "fake_sampler.sh"
    sampler.write_text("#!/bin/bash\necho $$ > \"$1\"\nexec sleep 600\n")
    sampler.chmod(0o755)

    # Spawn the fake sampler.
    proc = subprocess.Popen(["/bin/bash", str(sampler), str(pidfile)])
    # Wait for pidfile.
    for _ in range(50):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    pid_before = int(pidfile.read_text().strip())
    assert _pid_alive(pid_before), "fake sampler didn't start"

    # Kill-shard stub: kill the PID in pidfile.
    kill_sh = bin_dir / "kill_shard.sh"
    kill_sh.write_text(
        "#!/bin/bash\n"
        f'PID=$(cat "{pidfile}")\n'
        'kill -9 "$PID" 2>/dev/null || true\n'
        f'rm -f "{pidfile}"\n'
        'echo "killed shard 0 pid=$PID"\n'
    )
    kill_sh.chmod(0o755)

    # Launch-shards stub: respawn the fake sampler and write a marker.
    relaunched_marker = tmp_path / "relaunched.txt"
    launch_sh = bin_dir / "launch_shards.sh"
    launch_sh.write_text(
        "#!/bin/bash\n"
        f'echo "relaunched shard $1" > "{relaunched_marker}"\n'
        f'nohup bash "{sampler}" "{pidfile}" >/dev/null 2>&1 &\n'
        # Wait for pidfile so the test can deterministically check it.
        f'for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -s "{pidfile}" ]] && break; sleep 0.05; done\n'
    )
    launch_sh.chmod(0o755)

    # Train stub: fast no-op that touches an output marker.
    train_stub = bin_dir / "fake_train.sh"
    train_stub.write_text(
        "#!/bin/bash\n"
        '# args: --snapshot ... --output-dir ...\n'
        'OUT=""\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  case "$1" in\n'
        '    --output-dir) OUT="$2"; shift 2 ;;\n'
        '    *) shift ;;\n'
        '  esac\n'
        'done\n'
        'mkdir -p "$OUT"\n'
        'touch "$OUT/done"\n'
        'echo "fake train done -> $OUT"\n'
    )
    train_stub.chmod(0o755)

    # Eval stub: writes a fake results JSON with predictable fields.
    eval_stub = bin_dir / "fake_eval.sh"
    eval_stub.write_text(
        "#!/bin/bash\n"
        'OUT=""\n'
        'SUBSET=""\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  case "$1" in\n'
        '    --output) OUT="$2"; shift 2 ;;\n'
        '    --subset) SUBSET="$2"; shift 2 ;;\n'
        '    *) shift ;;\n'
        '  esac\n'
        'done\n'
        'mkdir -p "$(dirname "$OUT")"\n'
        'cat > "$OUT" <<EOF\n'
        '{\n'
        '  "model": "stub",\n'
        '  "lora": "stub-lora",\n'
        '  "n_items": '"${SUBSET:-200}"',\n'
        '  "subset_size": '"${SUBSET:-200}"',\n'
        '  "metrics": {"overall": {"acc": 0.42, "n": '"${SUBSET:-200}"', "wrong_format": 0},\n'
        '              "by_ui_type": {"text": {"acc": 0.50, "n": 100, "wrong_format": 0},\n'
        '                              "icon": {"acc": 0.34, "n": 100, "wrong_format": 0}}}\n'
        '}\n'
        'EOF\n'
        'echo "fake eval wrote $OUT"\n'
    )
    eval_stub.chmod(0o755)

    yield {
        "tmp_path": tmp_path,
        "samples": samples,
        "snapshots": snapshots,
        "adapters": adapters,
        "results": results,
        "bin_dir": bin_dir,
        "pidfile": pidfile,
        "pid_before": pid_before,
        "kill_sh": kill_sh,
        "launch_sh": launch_sh,
        "train_stub": train_stub,
        "eval_stub": eval_stub,
        "relaunched_marker": relaunched_marker,
        "proc": proc,
    }
    # Teardown: kill child in case it's still around.
    try:
        proc.kill()
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_train_run_sh_exists_and_executable():
    assert TRAIN_RUN_SH.is_file(), f"{TRAIN_RUN_SH} not found"
    mode = TRAIN_RUN_SH.stat().st_mode
    assert mode & stat.S_IXUSR, "train_run.sh is not executable"


def test_train_run_orchestrates_full_pipeline(stage):
    """End-to-end: kill, freeze, train, eval, leaderboard, relaunch."""
    import sys as _sys
    env = {
        **os.environ,
        "SAMPLES_DIR": str(stage["samples"]),
        "SNAPSHOTS_DIR": str(stage["snapshots"]),
        "ADAPTERS_DIR": str(stage["adapters"]),
        "RESULTS_DIR": str(stage["results"]),
        "KILL_SHARD_CMD": str(stage["kill_sh"]),
        "LAUNCH_SHARDS_CMD": str(stage["launch_sh"]),
        "TRAIN_LORA_CMD": str(stage["train_stub"]),
        "EVAL_CMD": str(stage["eval_stub"]),
        "SUBSET_SIZE": "200",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHON": _sys.executable,  # use the same interpreter pytest is running under
    }
    result = subprocess.run(
        ["/bin/bash", str(TRAIN_RUN_SH), "test"],
        env=env, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
    )
    assert result.returncode == 0, (
        f"train_run.sh failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # 1. Old shard 0 PID is gone. Reap the zombie first (the test was the
    # original parent of the fake-sampler process; on macOS os.kill(pid, 0)
    # returns success for a zombie until its parent waits on it).
    try:
        stage["proc"].wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    assert not _pid_alive(stage["pid_before"]), (
        f"shard 0 PID {stage['pid_before']} should have been killed"
    )

    # 2. Snapshot was frozen.
    snap = stage["snapshots"] / "run_test"
    assert snap.is_dir(), "snapshot dir should exist"
    assert (snap / "sft.jsonl").is_file()
    meta = json.loads((snap / "snapshot_meta.json").read_text())
    assert meta["run_id"] == "test"
    assert meta["total_sft_items"] == 2

    # 3. Train stub touched its done marker.
    adapter = stage["adapters"] / "lora_test"
    assert adapter.is_dir()
    assert (adapter / "done").is_file()

    # 4. Eval stub wrote a results JSON with subset_size.
    res_json = stage["results"] / "lora_test.json"
    assert res_json.is_file()
    payload = json.loads(res_json.read_text())
    assert payload["subset_size"] == 200

    # 5. Leaderboard was appended.
    leaderboard_txt = stage["results"] / "leaderboard.txt"
    leaderboard_json = stage["results"] / "leaderboard.json"
    assert leaderboard_txt.is_file()
    assert leaderboard_json.is_file()
    rows = json.loads(leaderboard_json.read_text())
    assert any(r["run_name"] == "lora_test" for r in rows)

    # 6. Sampler shard 0 was relaunched.
    assert stage["relaunched_marker"].is_file()
    assert "relaunched shard 0" in stage["relaunched_marker"].read_text()
    # And a new PID exists.
    new_pid = int(stage["pidfile"].read_text().strip())
    assert new_pid != stage["pid_before"]
    assert _pid_alive(new_pid), "new sampler should be alive"
    # Cleanup: kill the new fake.
    try:
        os.kill(new_pid, 9)
    except ProcessLookupError:
        pass
