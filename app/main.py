"""Entry point: python -m app.main --backend cpu --source demo"""

from __future__ import annotations

import argparse

from .decision import DefectDecider
from .frame_source import make_source
from .inference import make_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="i.MX93 edge vision defect detection")
    parser.add_argument("--backend", choices=["cpu", "npu", "heuristic"], default="cpu")
    parser.add_argument("--source", default="demo",
                        help="'demo', 'webcam', 'csi', or a path to an image folder")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    engine = make_engine(args.backend)
    decider = DefectDecider(threshold=args.threshold)
    print(f"[init] engine={engine.name} source={args.source}")

    defects = 0
    for i, frame in enumerate(make_source(args.source)):
        result = engine.infer(frame)
        alert = decider.update(result)
        if alert:
            defects += 1
        print(f"frame {i:04d}  {result.label:7s}  score={result.score:.2f}  "
              f"{result.latency_ms:6.2f} ms  {'<< DEFECT ALERT' if alert else ''}")
    print(f"[done] defect alerts: {defects}")


if __name__ == "__main__":
    main()
