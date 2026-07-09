# Benchmarks

Model: MobileNetV2 (α=1.0, 224×224), full-integer INT8, 2 classes — 2.7 MB
(`defect_int8.tflite`), 2.2 MB after Vela compilation (`defect_int8_vela.tflite`).

## Accuracy

| Stage | Val accuracy (60 images, synthetic dataset) |
|---|---|
| Float32 (SavedModel) | 100 % |
| INT8 full-integer PTQ | 100 % — no quantization loss |

## Vela compilation report (ethos-u65-256, Ethos_U65_High_End)

| Metric | Value |
|---|---|
| NPU operators | **94 / 94 (100 %)** — zero CPU fallback |
| CPU operators | 0 |
| Total SRAM used | 1 474 KiB |
| Total DRAM used | 2 348 KiB |
| MACs per inference | 299.6 M |

## Latency

Reproduce with `python models/benchmark.py --backend {cpu,npu} --runs 200`.

| Platform | Backend | Mean | p50 | p95 | Throughput |
|---|---|---|---|---|---|
| Host PC (x86-64, XNNPACK) | cpu | 3.53 ms | 3.38 ms | 4.33 ms | ~283 FPS |
| i.MX93 Cortex-A55 (2×1.7 GHz) | cpu | *TBD — run on board* | | | |
| i.MX93 Ethos-U65 (256 MACs/cc) | npu | *TBD — run on board* | | | |

> To fill in the board rows: copy the repo to the i.MX93 and run
> `python3 models/benchmark.py --backend cpu --runs 200` and
> `python3 models/benchmark.py --backend npu --runs 200`, then paste the output here.

Typical published numbers for MobileNetV2 INT8 on i.MX93 are ~3–5 ms on the
Ethos-U65 vs ~30–40 ms on the Cortex-A55 — roughly a **10× speed-up**; verify
with your own board and BSP.
