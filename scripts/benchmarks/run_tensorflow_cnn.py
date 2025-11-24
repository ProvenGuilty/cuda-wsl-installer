#!/usr/bin/env python3
"""TensorFlow CNN mini benchmark for CUDA WSL installer."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TF CNN training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)  # Already low
    parser.add_argument("--batch-size", type=int, default=128)  # Reduced from 256
    parser.add_argument("--result-file", type=Path)
    return parser.parse_args()


def configure_env(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


ARGS = parse_args()
configure_env(ARGS.device)

import tensorflow as tf  # noqa: E402


def ensure_device(device: str) -> None:
    if device == "cuda":
        print("Warning: TensorFlow GPU detection may not work, proceeding anyway")


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def load_data():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    return x_train, y_train


def run_benchmark() -> dict:
    ensure_device(ARGS.device)
    x_train, y_train = load_data()
    model = build_model()

    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=ARGS.epochs,
        batch_size=ARGS.batch_size,
        verbose=0,
    )
    duration = time.perf_counter() - start

    return {
        "device": ARGS.device,
        "epochs": ARGS.epochs,
        "batch_size": ARGS.batch_size,
        "seconds": duration,
        "history": {"loss": history.history["loss"]},
    }


def main() -> None:
    result = run_benchmark()
    print(
        f"[benchmark] tf_cnn device={result['device']} epochs={result['epochs']} "
        f"time={result['seconds']:.2f}s"
    )
    if ARGS.result_file:
        ARGS.result_file.parent.mkdir(parents=True, exist_ok=True)
        ARGS.result_file.write_text(json.dumps(result, indent=2))

    # Leaderboard integration
    import subprocess
    import os
    from datetime import datetime

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

    # Append to shared leaderboard file
    benchmark_type = "tensorflow_cnn"
    leaderboard_file = os.path.join(os.path.dirname(__file__), f"../../results/hacker_leaderboard_{benchmark_type}.json")
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            scores = json.load(f)
    else:
        scores = []

    # Add your new score (customize based on benchmark type)
    # Get system info for this run
    try:
        cpu_info = subprocess.check_output("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2", shell=True).decode().strip()
    except:
        cpu_info = "Unknown CPU"
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
    except:
        gpu_info = "Unknown GPU"
    try:
        cuda_version = "12.5"  # Installed CUDA version
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
        "benchmark": "tensorflow_cnn",
        "score": result['seconds'],
        "status": "ELITE HACKER!",  # Randomize or customize
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "os": os_info
    }
    scores.append(new_entry)

    # Sort by lowest score (best first) and keep top 100
    scores = sorted(scores, key=lambda x: x.get("score", float('inf')))[:100]

    with open(leaderboard_file, 'w') as f:
        json.dump(scores, f, indent=2)

    # Display the leaderboard
    print_hacker_leaderboard(scores)


if __name__ == "__main__":
    main()
