# i.MX93 Edge Vision Inspection

**Real-time visual defect detection on the NXP i.MX93, accelerated by the Arm Ethos-U65 NPU.**

An end-to-end Edge AI project: a quantized TFLite vision model is compiled with the Vela compiler and deployed on the i.MX93 NPU for on-device surface-defect classification, with a host **simulation mode** so the full pipeline can be developed and demoed without hardware.

> 🇩🇪 **Kurzbeschreibung:** Echtzeit-Fehlererkennung (Oberflächeninspektion) auf der NXP i.MX93 mit Ethos-U65 NPU. Quantisiertes TFLite-Modell, Vela-kompiliert, mit Simulationsmodus für die Entwicklung ohne Hardware.

## Why this project

Automated optical inspection (AOI) is a core Industry 4.0 use case: detecting scratches, dents, and contamination on parts directly at the production line — with low latency, no cloud dependency, and low power.

| | |
|---|---|
| **Target** | NXP i.MX93 (Cortex-A55 + Ethos-U65 NPU), NXP eIQ / Yocto Linux |
| **Model** | Quantized (INT8) CNN classifier, TensorFlow Lite, Vela-compiled for Ethos-U |
| **Pipeline** | Camera (GStreamer / V4L2) → preprocessing → NPU inference → decision + telemetry |
| **Sim mode** | Same pipeline on a host PC using image folders or webcam, CPU inference |

## Architecture

```
             ┌────────────────────────── i.MX93 (or host sim) ──────────────────────────┐
 Camera ───► │ FrameSource ──► Preprocess ──► InferenceEngine ──► DefectDecision ──► MQTT│──► Dashboard
 (CSI/USB)   │ (GStreamer /    (resize,       (TFLite +           (thresholds,          │    / logs
  or images  │  image folder)   normalize,     Ethos-U delegate    debounce,            │
             │                  INT8 quant)    or CPU fallback)    alerting)            │
             └──────────────────────────────────────────────────────────────────────────┘
```

## Quick start (simulation mode, no hardware needed)

```bash
git clone https://github.com/Feki-Tech/imx93-edge-vision-inspection.git
cd imx93-edge-vision-inspection
pip install -r app/requirements.txt
python -m app.main --backend cpu --source demo   # runs on bundled sample images
```

## Running on i.MX93

1. Flash an NXP eIQ-enabled Yocto image (Linux BSP with `ethosu` kernel driver and `tflite-runtime`).
2. Compile the model for the NPU: `./models/export_vela.sh`
3. Copy the repo to the board and run:

```bash
python3 -m app.main --backend npu --source csi
```

## Project status / roadmap

- [x] Repository scaffold, pipeline skeleton, CI
- [x] Training script (MobileNetV2 transfer learning) — see [docs/training.md](docs/training.md)
- [x] INT8 post-training quantization + Vela compilation (100 % NPU residency)
- [x] Trained INT8 model committed — 100 % val accuracy, no quantization loss ([docs/benchmarks.md](docs/benchmarks.md))
- [x] MQTT telemetry + live web dashboard ([docs/telemetry.md](docs/telemetry.md))
- [ ] GStreamer camera capture validated on i.MX93
- [ ] NPU vs CPU latency measured on the board ([docs/benchmarks.md](docs/benchmarks.md) has host numbers)
- [ ] Real production dataset (current model is trained on synthetic surfaces)
- [ ] Demo video

## Repository layout

```
app/        Inference application (runs on target and host)
models/     Training, quantization, Vela export, benchmark + committed INT8 models
dashboard/  Live web dashboard (MQTT over WebSockets, zero build step)
tests/      Unit tests (run in CI, no hardware required)
docs/       Architecture, training, deployment, telemetry and benchmark docs
```

## Documentation

- [Architecture](docs/architecture.md) — components, design decisions, model pipeline
- [Training guide](docs/training.md) — dataset, transfer learning, INT8 quantization, Vela
- [i.MX93 deployment](docs/deployment-imx93.md) — BSP requirements, running on the NPU, troubleshooting
- [Telemetry & dashboard](docs/telemetry.md) — MQTT topics, broker setup, live dashboard
- [Benchmarks](docs/benchmarks.md) — accuracy, Vela report, latency results

## License

MIT — see [LICENSE](LICENSE).
