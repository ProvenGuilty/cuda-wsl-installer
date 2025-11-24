#!/bin/bash
# Wipe CUDA installation and user data for clean testing

echo "Wiping CUDA system installation..."
sudo apt-get remove --purge cuda* nvidia* -y
sudo apt-get autoremove -y
sudo apt-get autoclean

echo "Removing user virtual environment and benchmark data..."
rm -rf ~/.cuda-wsl-bench-venv ~/.cuda-wsl-benchmarks

echo "Resetting environment variables..."
unset CUDA_HOME LD_LIBRARY_PATH

echo "Wipe complete. Restart WSL if needed: wsl --shutdown && wsl"
