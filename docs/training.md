# Training the defect model

## 1. Get a dataset

Two good options:

- **Your own photos**: capture "good" and "defect" images of a part/surface with any
  camera. Even 100–200 images per class works for transfer learning.
- **MVTec AD** (industry-standard anomaly detection benchmark,
  https://www.mvtec.com/company/research/datasets/mvtec-ad): pick one category
  (e.g. *metal_nut*), use `good` as-is and pool all defect types into `defect`.
  Free for non-commercial/research use.

Arrange it as:

```
data/
  train/
    good/    *.jpg
    defect/  *.jpg
  val/
    good/    *.jpg
    defect/  *.jpg
```

A common split is 80% train / 20% val per class.

## 2. Train (float32, transfer learning)

```sh
pip install -r models/requirements-train.txt
python models/train.py --data data --epochs 10
```

MobileNetV2 (ImageNet weights, frozen) + small classification head. On a laptop CPU
this takes minutes, not hours. Expect >95% val accuracy on single-category MVTec data.

## 3. Quantize to INT8

```sh
python models/quantize.py --saved-model models/saved_model --data data
```

Produces `models/defect_int8.tflite` — fully integer (weights *and* activations),
calibrated on ~200 representative training images. This step is mandatory:
the Ethos-U65 executes INT8 ops only.

Check accuracy after quantization — a drop of more than ~1–2% usually means the
representative dataset was too small or unrepresentative.

## 4. Compile for the NPU

```sh
pip install ethos-u-vela
./models/export_vela.sh models/defect_int8.tflite
```

Vela maps supported ops onto the Ethos-U65 (config `ethos-u65-256`) and prints a
report of which ops run on NPU vs CPU. Aim for 100% NPU residency; MobileNetV2 ops
are fully supported.

## 5. Use it

```sh
# Host (CPU): pip install tflite-runtime, then
python -m app.main --backend cpu --source data/val/defect

# Board (NPU):
python3 -m app.main --backend npu --source csi
```
