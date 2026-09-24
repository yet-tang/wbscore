#!/usr/bin/env bash
#
# One-shot deploy script for wbscore on a fresh VPS.
# Tested on Ubuntu 22.04 LTS, Hetzner CX22 (2 vCPU / 4GB).
#
# Usage (on your VPS as root):
#   curl -fsSL https://raw.githubusercontent.com/YOUR_USER/wbscore/main/deploy.sh | bash
#
# Or after git clone:
#   cd /opt/wbscore && bash deploy.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== wbscore deploy ==="
echo "Repo: $REPO_DIR"

# 1. Check prerequisites
command -v docker >/dev/null 2>&1 || {
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
}
command -v docker >/dev/null 2>&1 || { echo "Docker install failed"; exit 1; }

# 2. Verify Caddyfile has real domain
if grep -q "wbscore.example.com" Caddyfile; then
    echo ""
    echo "⚠️  Caddyfile still has placeholder domain 'wbscore.example.com'."
    echo "    Edit Caddyfile first:  vim Caddyfile"
    echo "    Then re-run this script."
    exit 1
fi

# 3. Pull latest images / build
echo ""
echo "Building image..."
docker compose build --pull

# 4. Start
echo ""
echo "Starting services..."
docker compose up -d

# 5. Wait for healthy
echo ""
echo "Waiting for API to be healthy..."
for i in $(seq 1 30); do
    if docker compose exec -T api curl -fsS http://127.0.0.1:8080/v1/healthz >/dev/null 2>&1; then
        echo "✓ API is up"
        break
    fi
    sleep 2
done

# 6. Show status
echo ""
echo "=== Service status ==="
docker compose ps
echo ""
echo "=== Logs (last 30 lines) ==="
docker compose logs --tail=30

echo ""
echo "=== Done ==="
DOMAIN=$(grep -E "^[a-z0-9.-]+\s*\{" Caddyfile | head -1 | awk '{print $1}')
echo "Your API should be live at: https://${DOMAIN}/app/"
echo "Try: curl https://${DOMAIN}/v1/healthz"