#!/usr/bin/env bash
# Launch one streaming-shard worker per GPU. Daemon-style: nohup + & per shard,
# pid recorded under <samples-root>/shard_<i>/pid for kill_shard.sh.
#
# Usage:
#   sample/launch_shards.sh \
#       --num-shards 4 \
#       --gpus 0,1,2,3 \
#       --input data/osatlas_hard_50k.jsonl \
#       --samples-root samples \
#       [--n 8] [--temperature 0.7] [--batch-size 64]
#
# Env hooks (mostly for tests):
#   WORKER_CMD     — override the per-shard worker invocation. The script will
#                    call: $WORKER_CMD <shard_dir> <shard> <num_shards>
#                    (in addition to the usual sample_northstar.py invocation
#                    when not set). Used by tests to plug in a stub.
#   PYTHON_BIN     — python interpreter (default: python3)
#
# On-disk side-effects per shard:
#   <samples-root>/shard_<i>/pid       (worker pid, for kill)
#   <samples-root>/shard_<i>/log.txt   (stdout + stderr)
#
set -euo pipefail

NUM_SHARDS=""
GPUS=""
INPUT=""
SAMPLES_ROOT="samples"
N=8
TEMPERATURE=0.7
BATCH_SIZE=64
EXTRA_ARGS=()

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKER_PY="$SCRIPT_DIR/sample_northstar.py"

usage() {
    cat <<EOF >&2
Usage: $0 --num-shards N --gpus 0,1,...,N-1 --input PATH [options]
Options:
  --num-shards N         Number of shards to launch (must equal #gpus).
  --gpus a,b,c,...       Comma-separated GPU ids, one per shard.
  --input PATH           Input JSONL of source items.
  --samples-root DIR     Root for per-shard dirs (default: samples).
  --n N                  Samples per item (default 8).
  --temperature T        Sampling temperature (default 0.7).
  --batch-size N         vLLM batch size (default 64).
  --                     Pass remaining args verbatim to the worker.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --input)      INPUT="$2"; shift 2 ;;
        --samples-root) SAMPLES_ROOT="$2"; shift 2 ;;
        --n)          N="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$NUM_SHARDS" || -z "$GPUS" || -z "$INPUT" ]]; then
    echo "missing required args" >&2
    usage
    exit 1
fi

# Split GPUs and validate count.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [[ "${#GPU_ARR[@]}" -ne "$NUM_SHARDS" ]]; then
    echo "ERROR: --gpus has ${#GPU_ARR[@]} entries but --num-shards=$NUM_SHARDS" >&2
    exit 2
fi

mkdir -p "$SAMPLES_ROOT"

for ((shard=0; shard<NUM_SHARDS; shard++)); do
    GPU="${GPU_ARR[$shard]}"
    SHARD_DIR="$SAMPLES_ROOT/shard_$shard"
    mkdir -p "$SHARD_DIR"
    LOG_FILE="$SHARD_DIR/log.txt"

    if [[ -n "${WORKER_CMD:-}" ]]; then
        # Test hook: stub takes (shard_dir, shard, num_shards).
        CUDA_VISIBLE_DEVICES="$GPU" \
            nohup bash -c "exec '$WORKER_CMD' '$SHARD_DIR' '$shard' '$NUM_SHARDS'" \
            >"$LOG_FILE" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$GPU" \
            nohup "$PYTHON_BIN" "$DEFAULT_WORKER_PY" \
                --input "$INPUT" \
                --shard-dir "$SHARD_DIR" \
                --shard "$shard" \
                --num-shards "$NUM_SHARDS" \
                --gpu-id "$GPU" \
                --n "$N" \
                --temperature "$TEMPERATURE" \
                --batch-size "$BATCH_SIZE" \
                "${EXTRA_ARGS[@]}" \
            >"$LOG_FILE" 2>&1 &
    fi
    LAUNCH_PID=$!

    # Best-effort: wait briefly for the worker to write its OWN pid to pid
    # file (workers must do this to support stable kill across restarts).
    # If the worker crashes immediately, fall back to recording our launch PID
    # so kill_shard.sh has SOMETHING to operate on.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if [[ -s "$SHARD_DIR/pid" ]]; then
            break
        fi
        sleep 0.1
    done
    if [[ ! -s "$SHARD_DIR/pid" ]]; then
        echo "$LAUNCH_PID" > "$SHARD_DIR/pid"
    fi

    echo "[launch] shard $shard on GPU $GPU -> pid=$(cat "$SHARD_DIR/pid") log=$LOG_FILE"
done

echo "[launch] $NUM_SHARDS shards launched. Tail logs with:"
echo "  tail -F $SAMPLES_ROOT/shard_*/log.txt"
echo "Status:  python sample/status.py --samples-root $SAMPLES_ROOT"
