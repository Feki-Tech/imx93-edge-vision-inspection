"""Defect decision logic: confidence threshold + debounce over consecutive frames."""

from __future__ import annotations

from collections import deque

from .inference import InferenceResult


class DefectDecider:
    def __init__(self, threshold: float = 0.6, debounce: int = 2):
        self.threshold = threshold
        self.window: deque[bool] = deque(maxlen=debounce)

    def update(self, result: InferenceResult) -> bool:
        """Returns True when a defect alert should be raised."""
        self.window.append(result.label == "defect" and result.score >= self.threshold)
        return len(self.window) == self.window.maxlen and all(self.window)
