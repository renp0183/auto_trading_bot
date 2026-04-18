#!/usr/bin/env bash
# =============================================================================
# setup_oracle.sh — One-shot setup for Regime Trader on Oracle Cloud
#
# Usage:
#   bash setup_oracle.sh [install_dir]
#
# Default install_dir: /opt/regime-trader
#
# Supports:
#   - Ubuntu 22.04 / 24.04  (apt)
#   - Oracle Linux 8 / 9    (dnf)
# =============================================================================
set -euo pipefail

INSTALL_DIR="${1:-/opt/regime-trader}"
SERVICE_NAME="regime-trader"
PYTHON_VERSION="3.11"

# Detect OS
if   [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID}"        # ubuntu | ol (Oracle Linux) | centos | rhel
else
    echo "ERROR: Cannot detect OS. Exiting." >&2
    exit 1
fi

echo "============================================================"
echo "  Regime Trader — Oracle Cloud Setup"
echo "  OS: ${PRETTY_NAME:-$OS_ID}"
echo "  Install dir: ${INSTALL_DIR}"
echo "============================================================"
echo ""

# ── 1. Install Python 3.11 and system dependencies ───────────────────────────
echo "[1/7] Installing system packages …"
if [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
    sudo apt-get update -q
    sudo apt-get install -y -q \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        git \
        curl \
        iptables-persistent
elif [[ "$OS_ID" == "ol" || "$OS_ID" == "rhel" || "$OS_ID" == "centos" ]]; then
    sudo dnf install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-devel \
        python3-pip \
        git \
        curl
else
    echo "WARNING: Unknown OS '$OS_ID'. Trying apt-get …"
    sudo apt-get update -q
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git curl
fi
echo "  System packages installed."

# ── 2. Create install directory ───────────────────────────────────────────────
echo "[2/7] Creating install directory ${INSTALL_DIR} …"
sudo mkdir -p "${INSTALL_DIR}"
# Detect the current (non-root) user
CURRENT_USER="${SUDO_USER:-$(whoami)}"
sudo chown -R "${CURRENT_USER}:${CURRENT_USER}" "${INSTALL_DIR}"

# ── 3. Copy project files ─────────────────────────────────────────────────────
echo "[3/7] Copying project files …"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

if [ "${PROJECT_DIR}" != "${INSTALL_DIR}" ]; then
    rsync -av --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='.git' --exclude='venv' \
              "${PROJECT_DIR}/" "${INSTALL_DIR}/"
    echo "  Files copied from ${PROJECT_DIR} to ${INSTALL_DIR}"
else
    echo "  Already in install dir — skipping copy."
fi

# ── 4. Create Python virtualenv and install dependencies ──────────────────────
echo "[4/7] Creating Python ${PYTHON_VERSION} virtualenv …"
cd "${INSTALL_DIR}"
python${PYTHON_VERSION} -m venv venv
echo "  Virtualenv created at ${INSTALL_DIR}/venv"

echo "  Installing Python dependencies (this may take a minute) …"
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
echo "  Dependencies installed."

# ── 5. Create required directories ───────────────────────────────────────────
echo "[5/7] Creating runtime directories …"
mkdir -p "${INSTALL_DIR}/logs"
mkdir -p "${INSTALL_DIR}/models"
echo "  logs/ and models/ created."

# ── 6. Install and enable systemd service ────────────────────────────────────
echo "[6/7] Installing systemd service …"

# Detect username for the service file
if id "ubuntu" &>/dev/null; then
    SERVICE_USER="ubuntu"
elif id "opc" &>/dev/null; then
    SERVICE_USER="opc"
else
    SERVICE_USER="${CURRENT_USER}"
fi

# Patch the service file with the correct user and install dir
SERVICE_SRC="${INSTALL_DIR}/deploy/regime-trader.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

sed "s|User=ubuntu|User=${SERVICE_USER}|g; \
     s|Group=ubuntu|Group=${SERVICE_USER}|g; \
     s|WorkingDirectory=/opt/regime-trader|WorkingDirectory=${INSTALL_DIR}|g; \
     s|EnvironmentFile=/opt/regime-trader|EnvironmentFile=${INSTALL_DIR}|g; \
     s|ExecStart=/opt/regime-trader|ExecStart=${INSTALL_DIR}|g" \
    "${SERVICE_SRC}" | sudo tee "${SERVICE_DST}" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
echo "  Service installed: ${SERVICE_DST}"
echo "  Service enabled for auto-start on boot."

# ── 7. Firewall — allow outbound HTTPS ───────────────────────────────────────
echo "[7/7] Configuring firewall (outbound HTTPS for Alpaca API) …"
if command -v iptables &>/dev/null; then
    # These rules are safe to run multiple times (they add if not present)
    sudo iptables -C OUTPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || \
        sudo iptables -I OUTPUT -p tcp --dport 443 -j ACCEPT
    sudo iptables -C OUTPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || \
        sudo iptables -I OUTPUT -p tcp --dport 80 -j ACCEPT
    # Persist rules
    if command -v netfilter-persistent &>/dev/null; then
        sudo netfilter-persistent save
    elif [ -d /etc/iptables ]; then
        sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null
    fi
    echo "  Outbound HTTPS (443) and HTTP (80) allowed."
else
    echo "  iptables not found — skipping firewall config."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Add your Alpaca credentials:"
echo "     cat > ${INSTALL_DIR}/.env << 'EOF'"
echo "     ALPACA_API_KEY=your_key_here"
echo "     ALPACA_SECRET_KEY=your_secret_here"
echo "     ALPACA_PAPER=true"
echo "     EOF"
echo "     chmod 600 ${INSTALL_DIR}/.env"
echo ""
echo "  2. Start the bot:"
echo "     sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "  3. Watch live logs:"
echo "     journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "  4. Check status:"
echo "     sudo systemctl status ${SERVICE_NAME}"
echo ""
echo "  The bot will auto-restart on crash and on server reboot."
echo ""
