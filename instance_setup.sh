#!/usr/bin/env bash
# Run AFTER gh auth login + huggingface-cli login on the Brev instance.
# Sets up the repo, installs deps, downloads small smoke data, runs end-to-end.

set -euxo pipefail

cd "$HOME"

# ---- 1. Clone (assumes gh auth done) ----
if [ ! -d click-distill ]; then
  gh repo clone gabrielnan/click-distill
fi
cd click-distill
git pull

# ---- 2. Install Python deps ----
# vLLM is the heavy one (~5min on first install)
pip install --upgrade pip
pip install -r requirements.txt
pip install vllm  # if not in requirements.txt

# ---- 3. Verify GPU + Northstar pull ----
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# Pre-pull Northstar weights to HF cache (background — overlap with data download)
huggingface-cli download Tzafon/Northstar-CUA-Fast --repo-type model &
HF_PULL_PID=$!

# ---- 4. Download small smoke data (annotations only, no images yet) ----
python data/download_osatlas.py --n 100 --output data/osatlas_hard_100.jsonl

# ---- 5. Wait for Northstar download ----
wait $HF_PULL_PID

# ---- 6. Download OS-Atlas image archives (just one, for smoke) ----
# Linux only = ~8GB, fastest. We don't need 110GB for smoke.
mkdir -p ~/osatlas_images/desktop_domain/linux_images
huggingface-cli download OS-Copilot/OS-Atlas-data \
    --repo-type dataset \
    --include "desktop_domain/linux_images.tar.7z" \
    --local-dir ~/osatlas_images_raw
# extract (need 7z installed)
which 7z || sudo apt-get install -y p7zip-full
cd ~/osatlas_images
7z x ~/osatlas_images_raw/desktop_domain/linux_images.tar.7z -y
tar -xf linux_images.tar -C desktop_domain/
rm -rf ~/osatlas_images_raw  # free disk
cd ~/click-distill

# ---- 7. Download ScreenSpot-Pro for eval ----
python eval/download_screenspot_pro.py

# ---- 8. SMOKE TEST: 1 shard, 100 items ----
echo "=== SMOKE TEST ==="
python sample/sample_northstar.py \
    --input data/osatlas_hard_100.jsonl \
    --output sample/raw_smoke.jsonl \
    --shard 0 --num-shards 1 \
    --shard-dir samples_smoke \
    --gpu-id 0 \
    --n 4 \
    --temperature 0.7 \
    --images-root ~/osatlas_images
# (if streaming, this writes to samples_smoke/shard_0/sft.jsonl directly)

python sample/status.py --samples-root samples_smoke

# Verify SFT data
echo "=== SFT data preview ==="
head -3 samples_smoke/shard_0/sft.jsonl 2>/dev/null || head -3 sample/raw_smoke.jsonl
wc -l samples_smoke/shard_0/sft.jsonl 2>/dev/null

echo "=== SMOKE PASSED — ready to scale up ==="
