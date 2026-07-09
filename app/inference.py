"""Inference backends: TFLite (CPU or Ethos-U NPU delegate) with a heuristic fallback.

The heuristic fallback lets the full pipeline run before a trained model exists,
so CI and the demo work from day one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = ("good", "defect")


@dataclass
class InferenceResult:
    label: str
    score: float  # confidence for the predicted label
    latency_ms: float


class HeuristicEngine:
    """Placeholder 'model': flags frames with strong dark regions as defects."""

    name = "heuristic"

    def infer(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        gray = frame.mean(axis=2)
        dark_ratio = float((gray < 80).mean())
        score = min(1.0, dark_ratio * 50)
        label = "defect" if score > 0.5 else "good"
        confidence = score if label == "defect" else 1 - score
        return InferenceResult(label, confidence, (time.perf_counter() - t0) * 1000)


class TFLiteEngine:
    """Runs a quantized TFLite model on CPU, or on the Ethos-U65 NPU via delegate."""

    def __init__(self, model_path: str, use_npu: bool = False):
        from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore

        delegates = []
        if use_npu:
            delegates.append(load_delegate("/usr/lib/libethosu_delegate.so"))
        self.interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.name = "npu" if use_npu else "cpu"

    def infer(self, frame: np.ndarray) -> InferenceResult:
        scale, zero_point = self.input.get("quantization", (0.0, 0))
        if scale:  # INT8 quantized input
            data = (frame / 255.0 / scale + zero_point).astype(self.input["dtype"])
        else:
            data = (frame / 255.0).astype(np.float32)
        self.interpreter.set_tensor(self.input["index"], data[None, ...])

        t0 = time.perf_counter()
        self.interpreter.invoke()
        latency = (time.perf_counter() - t0) * 1000

        out = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        o_scale, o_zp = self.output.get("quantization", (0.0, 0))
        if o_scale:
            out = (out - o_zp) * o_scale
        idx = int(out.argmax())
        return InferenceResult(LABELS[idx], float(out[idx]), latency)


def make_engine(backend: str, model_path: str = "models/defect_int8_vela.tflite"):
    if backend in ("cpu", "npu") and Path(model_path).exists():
        return TFLiteEngine(model_path, use_npu=backend == "npu")
    return HeuristicEngine()
