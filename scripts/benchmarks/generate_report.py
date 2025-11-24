#!/usr/bin/env python3
"""Aggregate CUDA benchmark results and update leaderboard."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

TESTS = {
    "pytorch_matmul": {
        "label": "PyTorch MatMul",
        "metric": "average_seconds",
        "unit": "s",
    },
    "tf_cnn": {
        "label": "TensorFlow CNN",
        "metric": "seconds",
        "unit": "s",
    },
    "cudf_groupby": {
        "label": "RAPIDS cuDF groupby",
        "metric": "seconds",
        "unit": "s",
    },
    "cuda_matmul": {
        "label": "CUDA MatMul",
        "metric": "seconds",
        "unit": "s",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark summary")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--bench-set", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpu", required=True)
    return parser.parse_args()


def load_results(folder: Path) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for name in TESTS:
        path = folder / f"{name}.json"
        if path.exists():
            payload[name] = json.loads(path.read_text())
    return payload


def compute_summary(baseline: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, cfg in TESTS.items():
        base_obj = baseline.get(key)
        after_obj = after.get(key)
        if not base_obj or not after_obj:
            continue
        metric = cfg["metric"]
        base_val = base_obj.get(metric)
        after_val = after_obj.get(metric)
        if base_val is None or after_val is None:
            continue
        improvement = ((base_val - after_val) / base_val * 100.0) if base_val else 0.0
        rows.append(
            {
                "test": key,
                "label": cfg["label"],
                "unit": cfg["unit"],
                "baseline": base_val,
                "after": after_val,
                "improvement_pct": improvement,
            }
        )
    return rows


def save_plot(results: List[Dict[str, Any]], output: Path) -> None:
    if not output:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting optional
        return
    labels = [row["label"] for row in results]
    baseline = [row["baseline"] for row in results]
    after = [row["after"] for row in results]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([val - width / 2 for val in x], baseline, width, label="CPU baseline")
    ax.bar([val + width / 2 for val in x], after, width, label="CUDA after")
    ax.set_ylabel("Seconds (lower is better)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("CUDA WSL benchmark comparison")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def update_leaderboard(results: List[Dict[str, Any]], args: argparse.Namespace, timestamp: str) -> None:
    lb_path = args.leaderboard
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| Timestamp | Host | GPU | Track | Bench Set | Test | Baseline (s) | CUDA (s) | Improvement % |\n"
    divider = "|---|---|---|---|---|---|---:|---:|---:|\n"
    if not lb_path.exists():
        lb_path.write_text(header + divider)
    lines = lb_path.read_text().splitlines()
    if len(lines) < 2:
        lines = [header.strip(), divider.strip()]
    for row in results:
        lines.append(
            f"| {timestamp} | {args.host} | {args.gpu} | {args.track} | {args.bench_set} | "
            f"{row['label']} | {row['baseline']:.4f} | {row['after']:.4f} | {row['improvement_pct']:.2f}% |"
        )
    lb_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    baseline = load_results(args.baseline)
    after = load_results(args.after)
    results = compute_summary(baseline, after)
    if not results:
        raise SystemExit("No overlapping benchmark results to compare")
    timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    summary = {
        "timestamp": timestamp,
        "host": args.host,
        "gpu": args.gpu,
        "track": args.track,
        "bench_set": args.bench_set,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    update_leaderboard(results, args, timestamp)
    save_plot(results, args.output_plot)
    for row in results:
        print(
            f"[report] {row['label']}: baseline={row['baseline']:.4f}s "
            f"cuda={row['after']:.4f}s improvement={row['improvement_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
