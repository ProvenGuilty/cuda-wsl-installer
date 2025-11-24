#!/usr/bin/env python3
"""RAPIDS cuDF vs pandas groupby benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Groupby benchmark")
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--result-file", type=Path)
    return parser.parse_args()


def run_cpu(rows: int) -> float:
    df = pd.DataFrame(
        {
            "key": np.random.randint(0, 1000, rows),
            "value": np.random.rand(rows),
        }
    )
    start = time.perf_counter()
    _ = df.groupby("key").value.mean()
    return time.perf_counter() - start


def run_gpu(rows: int) -> float:
    import cudf

    df = cudf.DataFrame(
        {
            "key": np.random.randint(0, 1000, rows),
            "value": np.random.rand(rows),
        }
    )
    start = time.perf_counter()
    _ = df.groupby("key").value.mean()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        try:
            import cudf  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"cudf import failed: {exc}") from exc

    if args.device == "cpu":
        duration = run_cpu(args.rows)
    else:
        duration = run_gpu(args.rows)

    payload = {
        "device": args.device,
        "rows": args.rows,
        "seconds": duration,
    }

    print(f"[benchmark] cudf_groupby device={args.device} rows={args.rows} time={duration:.2f}s")
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
