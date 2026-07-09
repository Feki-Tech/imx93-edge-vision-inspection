# Deploying on the i.MX93

Tested reference: NXP i.MX93 EVK / FRDM-IMX93 with an NXP eIQ-enabled Yocto Linux BSP.

## 1. Prepare the board

1. Flash a Linux BSP image that includes eIQ ML support (e.g. built with
   `imx-image-full` or a `meta-imx-ml` enabled build). Verify:
   ```sh
   ls /usr/lib/libethosu_delegate.so     # Ethos-U TFLite delegate
   ls /dev/ethosu0                       # NPU device node (ethosu kernel driver)
   python3 -c "import tflite_runtime"    # TFLite runtime
   ```
2. Connect a camera:
   - **CSI sensor** (e.g. OV5640 on the EVK): shows up as `/dev/video0`.
   - **USB webcam**: also `/dev/videoN`; check with `v4l2-ctl --list-devices`.

## 2. Build and copy the model

On your development PC:

```sh
pip install -r models/requirements-train.txt
python models/train.py --data data --epochs 10
python models/quantize.py --saved-model models/saved_model --data data
./models/export_vela.sh models/defect_int8.tflite
```

Copy the repo (with `models/defect_int8_vela.tflite`) to the board:

```sh
scp -r . root@<board-ip>:/opt/edge-vision/
```

## 3. Run

```sh
ssh root@<board-ip>
cd /opt/edge-vision
python3 -m app.main --backend npu --source csi
```

- `--backend npu` loads the Ethos-U delegate; use `--backend cpu` to compare latency.
- `--source csi` uses the GStreamer pipeline in `app/frame_source.py` — adjust
  resolution/device for your sensor.

## 4. Verify NPU offload

```sh
dmesg | grep ethosu          # driver activity
```

If the delegate fails to load, the most common causes are:

| Symptom | Fix |
|---|---|
| `libethosu_delegate.so not found` | BSP built without eIQ; add `meta-imx-ml` layer |
| Model runs but slowly on NPU | Model not Vela-compiled, or ops falling back to CPU — re-run `export_vela.sh` and check its op report |
| `/dev/ethosu0` missing | `ethosu` kernel module not loaded: `modprobe ethosu` |

## 5. Benchmarking

Run the same model with `--backend cpu` and `--backend npu` and compare the
`latency_ms` column printed per frame. Record results in `docs/benchmarks.md`.
