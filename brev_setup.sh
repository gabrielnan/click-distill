#!/usr/bin/env bash
set -euxo pipefail

# Clone repo
cd "$HOME"
if [ ! -d click-distill ]; then
  git clone https://github.com/gabrielnan/click-distill.git
fi
cd click-distill

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup done. Run 'huggingface-cli login' to paste HF token."
