#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[SKIP]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo "================================================"
echo "   Fraud Detection MLOps - Environment Setup"
echo "================================================"
echo ""

# ── Step 1: Swap ──────────────────────────────────────
echo "=== Step 1: Swap Setup ==="
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile && ok "Swap file created" || fail "fallocate failed"
    sudo chmod 600 /swapfile && ok "Permissions set"
    sudo mkswap /swapfile && ok "Formatted as swap"
    sudo swapon /swapfile && ok "Swap activated"
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
        ok "Swap added to /etc/fstab (permanent)"
    else
        warn "Swap already in /etc/fstab"
    fi
else
    warn "Swap file already exists at /swapfile"
    if ! swapon --show | grep -q /swapfile; then
        sudo swapon /swapfile && ok "Swap was off — turned on"
    else
        warn "Swap already active"
    fi
fi
free -h | grep -E "Mem|Swap"
echo ""

# ── Step 2: Project Directories ───────────────────────
echo "=== Step 2: Project Directories ==="
BASE="/home/$USER/Videos/mlops/assignment3"
DIRS=(
    "$BASE/data"
    "$BASE/notebooks"
    "$BASE/src"
    "$BASE/pipeline"
    "$BASE/cicd/.github/workflows"
    "$BASE/monitoring/grafana_dashboards"
    "$BASE/docker"
    "$BASE/tests"
    "$BASE/k8s"
    "$BASE/report"
    "/home/$USER/Videos/mlops/assignment3/fraud-artifacts"
    "/home/$USER/Videos/mlops/assignment3/mlruns"
)
for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" && ok "Created $dir"
    else
        warn "Already exists: $dir"
    fi
done
echo ""

# ── Step 3: Python dependencies ───────────────────────
echo "=== Step 3: Python Dependencies ==="
# Read requirements.txt, ignore comments and empty lines
while read -r pkg || [ -n "$pkg" ]; do
    # Skip comments and empty lines
    if [[ "$pkg" =~ ^#.*$ ]] || [[ -z "$pkg" ]]; then
        continue
    fi
    # Strip any carriage returns or extra spaces
    pkg=$(echo "$pkg" | tr -d '\r' | xargs)
    # Extract package name without version pinning
    pkg_name=$(echo "$pkg" | cut -d= -f1 | cut -d> -f1 | cut -d< -f1 | cut -d~ -f1)
    
    if pip show "$pkg_name" &>/dev/null; then
        warn "$pkg_name already installed"
    else
        pip install "$pkg" --break-system-packages && ok "Installed $pkg"
    fi
done < "$BASE/requirements.txt"
echo ""

# ── Final Summary ─────────────────────────────────────
echo "================================================"
echo "   Setup Complete — Next Steps"
echo "================================================"
echo ""
echo "1. Source environment variables if any:"
echo "   source ~/.bashrc"
echo ""
echo "2. Start the MLflow tracking server in a separate terminal:"
echo "   cd $BASE && mlflow ui --host 0.0.0.0 --port 5000"
echo "   then visit http://localhost:5000"
echo ""
echo "3. Run your MLflow pipeline:"
echo "   python3 pipeline/fraud_pipeline.py"
echo "================================================"
