#!/usr/bin/env bash
# train_run.sh — orchestrate one RSSD training run.
#
# Steps:
#   1. Kill sampler shard 0 (frees GPU 0 for train+eval).
#   2. Freeze the current samples/ snapshot into snapshots/run_<id>/.
#   3. LoRA SFT on the snapshot (writes adapters/lora_<id>/).
#   4. Subset-eval the new adapter (writes results/lora_<id>.json).
#   5. Append the result to results/leaderboard.{txt,json}.
#   6. Restart sampler shard 0 so the streaming pipeline keeps producing data.
#
# Usage:
#   ./train_run.sh <run_id>
#
# Env-var overrides (used by tests; safe defaults for production):
#   SAMPLES_DIR        default: samples
#   SNAPSHOTS_DIR      default: snapshots
#   ADAPTERS_DIR       default: adapters
#   RESULTS_DIR        default: results
#   KILL_SHARD_CMD     default: ./sample/kill_shard.sh
#   LAUNCH_SHARDS_CMD  default: ./sample/launch_shards.sh
#   TRAIN_LORA_CMD     default: python train/train_lora.py
#   EVAL_CMD           default: python eval/eval_screenspot_pro.py
#   SUBSET_SIZE        default: 200
#   GPU_ID             default: 0
#   BASE_MODEL         default: Tzafon/Northstar-CUA-Fast

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <run_id>" >&2
    exit 2
fi
RUN_ID="$1"

# Resolve repo root (dir containing this script).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAMPLES_DIR="${SAMPLES_DIR:-${REPO_ROOT}/samples}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-${REPO_ROOT}/snapshots}"
ADAPTERS_DIR="${ADAPTERS_DIR:-${REPO_ROOT}/adapters}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
KILL_SHARD_CMD="${KILL_SHARD_CMD:-${REPO_ROOT}/sample/kill_shard.sh}"
LAUNCH_SHARDS_CMD="${LAUNCH_SHARDS_CMD:-${REPO_ROOT}/sample/launch_shards.sh}"
PYTHON="${PYTHON:-python3}"
TRAIN_LORA_CMD="${TRAIN_LORA_CMD:-${PYTHON} ${REPO_ROOT}/train/train_lora.py}"
EVAL_CMD="${EVAL_CMD:-${PYTHON} ${REPO_ROOT}/eval/eval_screenspot_pro.py}"
SUBSET_SIZE="${SUBSET_SIZE:-200}"
GPU_ID="${GPU_ID:-0}"
BASE_MODEL="${BASE_MODEL:-Tzafon/Northstar-CUA-Fast}"

ADAPTER_OUT="${ADAPTERS_DIR}/lora_${RUN_ID}"
RESULT_JSON="${RESULTS_DIR}/lora_${RUN_ID}.json"
SNAPSHOT_DIR="${SNAPSHOTS_DIR}/run_${RUN_ID}"

mkdir -p "${SNAPSHOTS_DIR}" "${ADAPTERS_DIR}" "${RESULTS_DIR}"

echo "==> [1/6] kill sampler shard 0"
"${KILL_SHARD_CMD}" 0

echo "==> [2/6] freeze snapshot run_${RUN_ID}"
${PYTHON} "${REPO_ROOT}/train/freeze_snapshot.py" \
    --samples-dir "${SAMPLES_DIR}" \
    --snapshots-dir "${SNAPSHOTS_DIR}" \
    --run-id "${RUN_ID}"

# Pull total_sft_items out of the meta file so we can record it on the leaderboard.
TRAINED_ON=$(${PYTHON} -c "import json,sys; print(json.load(open('${SNAPSHOT_DIR}/snapshot_meta.json'))['total_sft_items'])")
echo "    snapshot has ${TRAINED_ON} sft rows"

echo "==> [3/6] LoRA SFT (gpu=${GPU_ID}) -> ${ADAPTER_OUT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" ${TRAIN_LORA_CMD} \
    --snapshot "${SNAPSHOT_DIR}/sft.jsonl" \
    --base-model "${BASE_MODEL}" \
    --output-dir "${ADAPTER_OUT}"

echo "==> [4/6] subset eval (n=${SUBSET_SIZE}) -> ${RESULT_JSON}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" ${EVAL_CMD} \
    --model "${BASE_MODEL}" \
    --lora "${ADAPTER_OUT}" \
    --subset "${SUBSET_SIZE}" \
    --output "${RESULT_JSON}"

echo "==> [5/6] append leaderboard"
PYTHONPATH="${REPO_ROOT}" ${PYTHON} - <<EOF
import json
from pathlib import Path
import sys
sys.path.insert(0, "${REPO_ROOT}")
from eval.leaderboard import append_result

payload = json.loads(Path("${RESULT_JSON}").read_text())
metrics = payload.get("metrics", {}) or {}
overall_acc = (metrics.get("overall") or {}).get("acc", 0.0)
text_acc = ((metrics.get("by_ui_type") or {}).get("text") or {}).get("acc", 0.0)
icon_acc = ((metrics.get("by_ui_type") or {}).get("icon") or {}).get("acc", 0.0)
result = {
    "overall_acc": overall_acc,
    "text_acc": text_acc,
    "icon_acc": icon_acc,
    "subset_size": payload.get("subset_size") or ${SUBSET_SIZE},
    "trained_on": ${TRAINED_ON},
}
append_result(Path("${RESULTS_DIR}"), "lora_${RUN_ID}", result)
print(f"  appended lora_${RUN_ID} (overall={overall_acc:.3f}, n={result['subset_size']})")
EOF

echo "==> [6/6] relaunch sampler shard 0"
"${LAUNCH_SHARDS_CMD}" 0

echo "[done] run_${RUN_ID} complete. Adapter: ${ADAPTER_OUT}  Result: ${RESULT_JSON}"
