"""Generate a synthetic surface-inspection dataset (good / defect).

Simulates brushed-metal-like surfaces; defect images get scratches, dents or
stains. Useful for developing the full train -> quantize -> deploy pipeline
before real production data is available.

Usage:
    python models/make_dataset.py --out data --per-class 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SIZE = 224


def make_surface(rng: np.random.Generator) -> np.ndarray:
    """Brushed-metal-like texture: gray base + noise + horizontal motion blur."""
    base = rng.integers(150, 200)
    img = np.full((SIZE, SIZE), base, dtype=np.float32)
    img += rng.normal(0, 12, img.shape)
    k = np.zeros((1, 15), dtype=np.float32)
    k[0, :] = 1 / 15
    img = cv2.filter2D(img, -1, k)  # horizontal brushing
    # slight illumination gradient
    gx = np.linspace(rng.uniform(-15, 15), rng.uniform(-15, 15), SIZE)
    img += gx[None, :]
    img = img.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def add_scratch(img: np.ndarray, rng: np.random.Generator) -> None:
    p1 = (int(rng.integers(10, SIZE - 10)), int(rng.integers(10, SIZE - 10)))
    angle = rng.uniform(0, np.pi)
    length = rng.integers(60, 180)
    p2 = (int(p1[0] + length * np.cos(angle)), int(p1[1] + length * np.sin(angle)))
    shade = int(rng.integers(20, 80))
    cv2.line(img, p1, p2, (shade, shade, shade), int(rng.integers(1, 4)))


def add_dent(img: np.ndarray, rng: np.random.Generator) -> None:
    c = (int(rng.integers(30, SIZE - 30)), int(rng.integers(30, SIZE - 30)))
    r = int(rng.integers(8, 25))
    shade = int(rng.integers(30, 100))
    overlay = img.copy()
    cv2.circle(overlay, c, r, (shade, shade, shade), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


def add_stain(img: np.ndarray, rng: np.random.Generator) -> None:
    c = (int(rng.integers(30, SIZE - 30)), int(rng.integers(30, SIZE - 30)))
    axes = (int(rng.integers(15, 45)), int(rng.integers(10, 30)))
    shade = int(rng.integers(60, 120))
    overlay = img.copy()
    cv2.ellipse(overlay, c, axes, float(rng.uniform(0, 180)), 0, 360,
                (shade, shade, int(shade * 0.8)), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


DEFECTS = (add_scratch, add_dent, add_stain)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_val = int(args.per_class * args.val_split)

    for label in ("good", "defect"):
        for i in range(args.per_class):
            img = make_surface(rng)
            if label == "defect":
                for fn in rng.choice(DEFECTS, size=rng.integers(1, 3), replace=True):
                    fn(img, rng)
            split = "val" if i < n_val else "train"
            d = Path(args.out) / split / label
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{label}_{i:04d}.jpg"), img)

    print(f"[dataset] {args.per_class} images/class written to {args.out}/ "
          f"(train/val split {1 - args.val_split:.0%}/{args.val_split:.0%})")


if __name__ == "__main__":
    main()
