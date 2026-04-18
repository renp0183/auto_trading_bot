#!/usr/bin/env bash
# =============================================================================
# update.sh — Pull latest code and restart the service cleanly.
#
# Usage (from the project directory on the Oracle server):
#   bash deploy/update.sh
# =============================================================================
set -euo pipefail

SERVICE_NAME="regime-trader"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "  Regime Trader — Update & Restart"
echo "============================================================"

cd "${INSTALL_DIR}"

# Pull latest code (only if this is a git repo)
if [ -d .git ]; then
    echo "[1/3] Pulling latest code …"
    git pull
else
    echo "[1/3] Not a git repo — skipping pull."
fi

# Reinstall dependencies (handles new/updated packages)
echo "[2/3] Updating Python dependencies …"
venv/bin/pip install -r requirements.txt -q

# Graceful restart: systemd sends SIGTERM, bot saves snapshot, then restarts
echo "[3/3] Restarting service …"
sudo systemctl restart "${SERVICE_NAME}"

echo ""
echo "Update complete. Watching logs (Ctrl+C to stop watching):"
echo ""
sleep 2
journalctl -u "${SERVICE_NAME}" -n 30 -f
