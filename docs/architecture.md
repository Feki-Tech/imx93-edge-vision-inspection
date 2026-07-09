# Architecture

## Overview

The system performs real-time surface-defect classification at the edge. All inference
runs on-device (i.MX93); only lightweight results/telemetry leave the device.

```
             ┌────────────────────────── i.MX93 (or host sim) ──────────────────────────┐
 Camera ───► │ FrameSource ──► Preprocess ──► InferenceEngine ──► DefectDecision ──► out │
 (CSI/USB)   │ (GStreamer /    (resize,       (TFLite +           (threshold +          │
  or images  │  image folder)   INT8 quant)    Ethos-U delegate)   debounce)            │
             └──────────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | File | Responsibility |
|---|---|---|
| FrameSource | `app/frame_source.py` | Uniform frame iterator over demo images, image folders, webcam, or i.MX93 CSI camera (GStreamer) |
| InferenceEngine | `app/inference.py` | TFLite inference on CPU or Ethos-U65 NPU (via `libethosu_delegate.so`); INT8 input/output quantization handling; heuristic fallback when no model is present |
| DefectDecider | `app/decision.py` | Confidence threshold + N-frame debounce to suppress single-frame false positives |
| Entry point | `app/main.py` | CLI wiring: `--backend {cpu,npu,heuristic} --source {demo,webcam,csi,<folder>}` |

## Design decisions

- **Same code on host and target.** The pipeline is pure Python + TFLite; only the
  frame source and delegate differ. This makes the project fully demoable and CI-testable
  without hardware.
- **Heuristic fallback engine.** Before a trained model exists, a simple dark-region
  heuristic stands in for the model. The pipeline, tests, and CI work from day one, and
  the interface (`InferenceResult`) is identical.
- **Full-integer INT8.** The Ethos-U65 only executes integer ops. Post-training
  quantization with a representative dataset (`models/quantize.py`) produces a fully
  INT8 model, which Vela then compiles into NPU command streams.
- **Debounced decisions.** A defect alert requires N consecutive high-confidence defect
  frames — a cheap, effective false-positive filter for line-scan scenarios.

## Model pipeline

```
train.py (MobileNetV2 transfer learning, float32)
   └─► SavedModel
quantize.py (post-training full-integer quantization, representative dataset)
   └─► defect_int8.tflite
export_vela.sh (Vela compiler, ethos-u65-256)
   └─► defect_int8_vela.tflite   ← deployed to the board
```

At runtime the Ethos-U delegate executes Vela-compiled subgraphs on the NPU;
unsupported ops (if any) fall back to the Cortex-A55 CPU via XNNPACK.
