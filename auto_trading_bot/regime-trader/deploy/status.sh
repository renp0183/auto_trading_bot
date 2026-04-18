#!/usr/bin/env bash
# =============================================================================
# status.sh — Quick health check without SSH-ing into each component manually.
# =============================================================================
SERVICE_NAME="regime-trader"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "  Regime Trader — Status"
echo "============================================================"
echo ""

# Service status
echo "── systemd service ──────────────────────────────────────────"
systemctl status "${SERVICE_NAME}" --no-pager -l 2>/dev/null || \
    echo "  Service not found — is it installed?"
echo ""

# State snapshot
SNAP="${INSTALL_DIR}/state_snapshot.json"
echo "── State snapshot ───────────────────────────────────────────"
if [ -f "${SNAP}" ]; then
    python3 -c "
import json, sys
with open('${SNAP}') as f:
    s = json.load(f)
fields = ['regime_label','regime_probability','equity','daily_pnl',
          'circuit_breaker','bars_processed','orders_submitted','last_bar_time']
for k in fields:
    v = s.get(k, 'N/A')
    print(f'  {k:<25s} {v}')
"
else
    echo "  No snapshot found — bot may not have run yet."
fi
echo ""

# Recent log tail
echo "── Last 20 log lines ────────────────────────────────────────"
journalctl -u "${SERVICE_NAME}" -n 20 --no-pager 2>/dev/null || \
    echo "  journald not available."
echo ""

# Check for halt lock
LOCK="${INSTALL_DIR}/trading_halted.lock"
if [ -f "${LOCK}" ]; then
    echo "⚠  WARNING: trading_halted.lock FILE EXISTS"
    echo "   Trading is halted due to peak drawdown."
    echo "   Review the situation, then: rm ${LOCK}"
    echo "   Then restart:  sudo systemctl restart ${SERVICE_NAME}"
    echo ""
fi
