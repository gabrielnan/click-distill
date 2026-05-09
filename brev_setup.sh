#!/usr/bin/env bash
set -euxo pipefail

# Clone (now public)
cd "$HOME"
if [ ! -d click-distill ]; then
  git clone https://github.com/gabrielnan/click-distill.git
fi
cd click-distill

# Deps
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup done. Next: huggingface-cli login (paste HF token)."
