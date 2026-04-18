#!/usr/bin/env bash
# =============================================================================
# start.sh — Manual start wrapper for debugging / interactive use.
#
# For production 24/7 operation use systemd instead:
#   sudo systemctl start regime-trader
#
# Usage:
#   bash deploy/start.sh [extra args passed to main.py live]
#
# Example (dry run):
#   bash deploy/start.sh --dry-run
# =============================================================================
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Working directory: ${INSTALL_DIR}"

cd "${INSTALL_DIR}"

if [ ! -f venv/bin/python ]; then
    echo "ERROR: virtualenv not found at ${INSTALL_DIR}/venv"
    echo "Run deploy/setup_oracle.sh first."
    exit 1
fi

if [ ! -f .env ]; then
    echo "WARNING: .env file not found — credentials must be in environment variables."
fi

echo "Starting Regime Trader …"
exec venv/bin/python main.py live "$@"
