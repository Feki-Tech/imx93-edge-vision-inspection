"""Frame sources: demo image generator, image folder, webcam, or CSI camera."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

import numpy as np

FRAME_SIZE = (224, 224)  # (width, height) expected by the model


def demo_frames(count: int = 20, seed: int = 42) -> Iterator[np.ndarray]:
    """Synthetic frames: uniform 'good' surfaces, some with a scratch-like defect."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        frame = np.full((*FRAME_SIZE[::-1], 3), 180, dtype=np.int16)
        frame += rng.integers(-10, 10, frame.shape, dtype=np.int16)
        frame = frame.clip(0, 255).astype(np.uint8)
        if i % 5 in (0, 1):  # inject defects in consecutive pairs
            x = int(rng.integers(20, FRAME_SIZE[0] - 20))
            frame[30:190, x : x + 8] = 40  # dark scratch
        yield frame


def folder_frames(path: str) -> Iterator[np.ndarray]:
    import cv2

    files = sorted(itertools.chain.from_iterable(Path(path).glob(p) for p in ("*.png", "*.jpg", "*.jpeg")))
    if not files:
        raise FileNotFoundError(f"No images found in {path}")
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            yield cv2.resize(img, FRAME_SIZE)


def camera_frames(device: int | str = 0) -> Iterator[np.ndarray]:
    """Webcam on host; on i.MX93 pass a GStreamer CSI pipeline string."""
    import cv2

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {device}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.resize(frame, FRAME_SIZE)
    finally:
        cap.release()


def make_source(name: str) -> Iterator[np.ndarray]:
    if name == "demo":
        return demo_frames()
    if name == "webcam":
        return camera_frames(0)
    if name == "csi":
        # i.MX93 CSI camera via GStreamer (adjust to your sensor/BSP)
        pipeline = (
            "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! "
            "videoconvert ! appsink"
        )
        return camera_frames(pipeline)
    return folder_frames(name)  # treat as image folder path
