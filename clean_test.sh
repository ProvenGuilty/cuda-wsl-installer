#!/bin/bash

# CUDA WSL Installer - Clean Test Script
# This script wipes the local repo and does a fresh install for testing

set -e  # Exit on any error

echo "🧹 CUDA WSL Installer - Clean Test Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -d "../CascadeProjects" ]]; then
    log_error "This script should be run from ~/CascadeProjects/cuda-wsl-installer/"
    exit 1
fi

cd ../CascadeProjects

# Remove existing repo if it exists
if [[ -d "cuda-wsl-installer" ]]; then
    log_info "Removing existing cuda-wsl-installer directory..."
    rm -rf cuda-wsl-installer
    log_success "Directory removed"
fi

# Clone fresh repo
log_info "Cloning fresh repository..."
if command -v git &> /dev/null && git --version &> /dev/null; then
    # Try SSH first, fallback to HTTPS
    if git clone git@github.com:ProvenGuilty/cuda-wsl-installer.git 2>/dev/null; then
        log_success "Cloned via SSH"
    else
        log_info "SSH clone failed, trying HTTPS..."
        git clone https://github.com/ProvenGuilty/cuda-wsl-installer.git
        log_success "Cloned via HTTPS"
    fi
else
    log_error "Git not found!"
    exit 1
fi

cd cuda-wsl-installer

# Pull latest changes
log_info "Pulling latest changes..."
git pull origin master
log_success "Repository updated"

# Optional: Run install
if [[ "$1" == "--install" ]]; then
    log_info "Running full install..."
    ./install.sh
elif [[ "$1" == "--dry-run" ]]; then
    log_info "Running dry-run..."
    ./install.sh --dry-run
else
    log_info "Ready for manual testing. Run:"
    echo "  cd cuda-wsl-installer"
    echo "  ./install.sh --dry-run    # Test dry-run"
    echo "  ./install.sh              # Full install"
fi

log_success "Clean test setup complete! 🎉"
