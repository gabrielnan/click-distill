#!/usr/bin/env bash
set -euxo pipefail

# Minimal prep. Repo clone + deps post-shell (private repo needs gh auth).
pip install --upgrade pip

# Ensure gh CLI present
if ! command -v gh >/dev/null 2>&1; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
  sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list
  sudo apt update
  sudo apt install -y gh
fi

echo "Setup done. Next:"
echo "  gh auth login"
echo "  git clone https://github.com/gabrielnan/click-distill.git"
echo "  cd click-distill && pip install -r requirements.txt"
echo "  huggingface-cli login"
