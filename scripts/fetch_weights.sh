#!/usr/bin/env bash
#
# Fetch the v4 model checkpoint from Hugging Face Hub.
# Required because model weights aren't in git (too large for GitHub's 100MB limit).
#
# Usage:
#   bash scripts/fetch_weights.sh
#
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$WORKSPACE/checkpoints/wb_laya_v4"
HF_REPO="yet-tang/wb-quality-laya-v4"  # adjust after upload

mkdir -p "$WEIGHTS_DIR"

if [ -f "$WEIGHTS_DIR/wb_laya.pt" ]; then
    echo "[skip] $WEIGHTS_DIR/wb_laya.pt already exists ($(du -h "$WEIGHTS_DIR/wb_laya.pt" | cut -f1))"
    exit 0
fi

echo "Downloading v4 weights from Hugging Face..."
python3 - <<EOF
from huggingface_hub import hf_hub_download
import os
os.makedirs("$WEIGHTS_DIR", exist_ok=True)
path = hf_hub_download(
    repo_id="$HF_REPO",
    filename="wb_laya.pt",
    local_dir="$WEIGHTS_DIR",
)
print(f"Saved to: {path}")
EOF

echo "Done. v4 weights ready at $WEIGHTS_DIR/wb_laya.pt"