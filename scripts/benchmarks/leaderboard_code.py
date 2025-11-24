# Add this after your existing result saving/printing code

import json
import os
from datetime import datetime
import sys

# Define the leaderboard display function
def print_hacker_leaderboard(scores):
    header = """
   ███╗░░██╗██╗░░░██╗██╗██████╗░██╗░█████╗░
   ████╗░██║██║░░░██║██║██╔══██╗██║██╔══██╗
   ██╔██╗██║██║░░░██║██║██║░░██║██║███████║
   ██║╚████║╚██╗░██╔╝██║██║░░██║██║██╔══██║
   ██║░╚███║░╚████╔╝░██║██████╔╝██║██║░░██║
   ╚═╝░░╚══╝░░╚═══╝░░╚═╝╚═════╝░╚═╝╚═╝░░╚═╝
═══════════════════════════════════════════════════════════════════════════════
║   PHREAKERS & HACKERZ CUDA WSL LEADERBOARD - BBS 1985 STYLE!                              ║
║   Scoring: Lower times = BETTER! (CUDA vs CPU battles, fastest wins!)                     ║
═══════════════════════════════════════════════════════════════════════════════════════════════
║ Rank │ Handle              │ Benchmark             │ Score      │ Status                 ║
╠══════╬═════════════════════╬══════════════════════╬════════════╬════════════════════════╣
"""
    footer = """
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
   ▀▄▀▄▀▄ YOU DA MAN! ▄▀▄▀▄   STAY HACKIN' - NO LAMERS ALLOWED   ▀▄▀▄▀▄ YOU DA MAN! ▀▄▀▄▀▄

System Specs for Top Scores (CPU vs GPU details):
"""

    print(header)
    for i, score in enumerate(scores[:10]):  # Show top 10
        rank = f"{i+1:2d}."
        handle = score.get('handle', 'Anonymous')[:19].ljust(19)
        benchmark = score['benchmark'][:20].ljust(20)
        time_score = f"{score['score']:.4f}s" if 'score' in score else score.get('time', 'DNF')
        status = score.get('status', 'UNKNOWN!')[:22].ljust(22)
        print(f"║ {rank}  │ {handle} │ {benchmark} │ {time_score} │ {status} ║")
    print(footer)
    
    # Detailed specs below
    for i, score in enumerate(scores[:5]):  # Details for top 5
        rank = i+1
        handle = score.get('handle', 'Anonymous')
        benchmark = score['benchmark']
        cpu = score.get('cpu', 'Unknown CPU')
        gpu = score.get('gpu', 'Unknown GPU')
        os_ = score.get('os', 'Unknown OS')
        cuda = score.get('cuda_version', 'Unknown CUDA')
        driver = score.get('driver_version', 'Unknown Driver')
        device_type = 'GPU' if 'cuda' in benchmark.lower() else 'CPU'
        print(f"{rank}. {handle} - {benchmark} ({device_type}): CPU: {cpu} | GPU: {gpu} | OS: {os_} | CUDA: {cuda} | Driver: {driver}")

# Append to shared leaderboard file (now in repo for public visibility)
if "pytorch" in __file__:
    benchmark_type = "pytorch_matmul"
elif "tensorflow" in __file__:
    benchmark_type = "tensorflow_cnn"
elif "cudf" in __file__:
    benchmark_type = "cudf_groupby"
else:
    benchmark_type = "unknown"
leaderboard_file = os.path.join(os.path.dirname(__file__), f"../../results/hacker_leaderboard_{benchmark_type}.json")
if os.path.exists(leaderboard_file):
    with open(leaderboard_file, 'r') as f:
        scores = json.load(f)
else:
    scores = []

# Add your new score (customize based on benchmark type)
# Get system info for this run
import subprocess
try:
    cpu_info = subprocess.check_output("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2", shell=True).decode().strip()
except:
    cpu_info = "Unknown CPU"
try:
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
except:
    gpu_info = "Unknown GPU"
try:
    cuda_version = subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
except:
    cuda_version = "Unknown CUDA"
try:
    os_info = subprocess.check_output("lsb_release -d | cut -f2", shell=True).decode().strip()
except:
    os_info = "Unknown OS"
try:
    driver_version = subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
except:
    driver_version = "Unknown Driver"

# Get GitHub handle from git config
try:
    github_handle = subprocess.check_output("git config user.name", shell=True).decode().strip()
    if not github_handle.startswith('@'):
        github_handle = f"@{github_handle}"
except:
    github_handle = "@Anonymous"

new_entry = {
    "handle": github_handle,
    "benchmark": "pytorch_matmul" if "pytorch" in __file__ else "tensorflow_cnn",
    "score": results["avg_time"] if "pytorch" in __file__ else results["total_duration"],
    "status": "ELITE HACKER!",  # Randomize or customize
    "cpu": cpu_info,
    "gpu": gpu_info,
    "cuda_version": cuda_version,
    "driver_version": driver_version,
    "os": os_info
}
    # Check if user already has a score for this benchmark, replace if so
    existing_index = next((i for i, s in enumerate(scores) if s.get('handle') == github_handle), None)
    if existing_index is not None:
        scores[existing_index] = new_entry
    else:
        scores.append(new_entry)

# Sort by lowest score (best first) and keep top 100
scores = sorted(scores, key=lambda x: x.get("score", float('inf')))[:100]

with open(leaderboard_file, 'w') as f:
    json.dump(scores, f, indent=2)

# Generate markdown for GitHub
try:
    subprocess.run([sys.executable, os.path.join(os.path.dirname(leaderboard_file), "generate_leaderboard_md.py")], check=True)
except:
    pass  # Ignore if generation fails

# Display the leaderboard
print_hacker_leaderboard(scores)
