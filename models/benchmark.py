"""Latency benchmark: runs the model N times and reports statistics.

Usage:
    python models/benchmark.py --backend cpu --runs 200
    python3 models/benchmark.py --backend npu --runs 200      # on i.MX93
"""

from __future__ import annotations

import argparse
import statistics

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.frame_source import demo_frames  # noqa: E402
from app.inference import make_engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "npu", "heuristic"], default="cpu")
    parser.add_argument("--model", default=None,
                        help="Model path (default: chosen automatically per backend)")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    engine = make_engine(args.backend, model_path=args.model)
    frames = list(demo_frames(count=20))

    for i in range(args.warmup):
        engine.infer(frames[i % len(frames)])

    lat = [engine.infer(frames[i % len(frames)]).latency_ms for i in range(args.runs)]
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95) - 1]
    print(f"engine={engine.name}  runs={args.runs}")
    print(f"latency  mean={statistics.mean(lat):.2f} ms  p50={p50:.2f} ms  "
          f"p95={p95:.2f} ms  min={lat[0]:.2f} ms  max={lat[-1]:.2f} ms")
    print(f"throughput ~ {1000 / statistics.mean(lat):.0f} FPS (single-threaded)")


if __name__ == "__main__":
    main()
