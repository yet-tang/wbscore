#!/usr/bin/env bash
#
# Fetch the v5 model checkpoint + laya-multilingual encoder from Hugging Face Hub.
# Required because model weights aren't in git (too large for GitHub's 100MB limit).
#
# v5 needs TWO things:
#   1. The trained head: checkpoints/wb_laya_v5/wb_laya.pt (~1.3GB)
#   2. The laya-multilingual encoder: ../laya_checkpoints/multilingual/encoder/ (~600MB)
#
# Usage:
#   bash scripts/fetch_weights.sh
#
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$WORKSPACE/checkpoints/wb_laya_v5"
HF_REPO_V5="yet-tang/wb-quality-laya-v5"
HF_REPO_ENCODER="convaiinnovations/laya-multilingual"

mkdir -p "$WEIGHTS_DIR"

# 1. Download v5 trained head
if [ ! -f "$WEIGHTS_DIR/wb_laya.pt" ]; then
    echo "[v5 head] downloading from Hugging Face..."
    python3 - <<EOF
from huggingface_hub import hf_hub_download
import os
os.makedirs("$WEIGHTS_DIR", exist_ok=True)
path = hf_hub_download(
    repo_id="$HF_REPO_V5",
    filename="wb_laya.pt",
    local_dir="$WEIGHTS_DIR",
)
print(f"  saved → {path}")
EOF
else
    echo "[skip] v5 head already at $WEIGHTS_DIR/wb_laya.pt ($(du -h "$WEIGHTS_DIR/wb_laya.pt" | cut -f1))"
fi

# 2. Download laya-multilingual encoder (the 322M backbone)
LAYA_LOCAL="$WORKSPACE/../laya_checkpoints/multilingual"
if [ ! -d "$LAYA_LOCAL/encoder" ]; then
    echo "[encoder] downloading laya-multilingual from Hugging Face..."
    python3 - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$HF_REPO_ENCODER",
    local_dir="$LAYA_LOCAL",
)
print(f"  saved → {path}")
EOF
else
    echo "[skip] encoder already at $LAYA_LOCAL/encoder"
fi

echo ""
echo "Done. v5 weights ready at $WEIGHTS_DIR/wb_laya.pt"
echo "Encoder ready at $LAYA_LOCAL/encoder/"
