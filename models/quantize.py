"""Post-training INT8 quantization for Ethos-U65 deployment.

Full-integer quantization is REQUIRED for the Ethos-U65 NPU: all weights and
activations must be INT8, with a representative dataset for calibration.

Usage:
    python models/quantize.py --saved-model models/saved_model --data data
    ./models/export_vela.sh models/defect_int8.tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def representative_dataset(data_dir: str, samples: int = 200):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train", image_size=IMG_SIZE, batch_size=1, label_mode=None, shuffle=True
    )

    def gen():
        for i, batch in enumerate(ds):
            if i >= samples:
                break
            yield [tf.cast(batch, tf.float32)]

    return gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model", default="models/saved_model")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="models/defect_int8.tflite")
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(args.data)
    # Full-integer quantization — mandatory for Ethos-U
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    Path(args.out).write_bytes(tflite_model)
    print(f"[quantize] INT8 model written to {args.out} "
          f"({len(tflite_model) / 1024:.0f} KiB)")

    # Sanity check: run one inference on CPU
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=inp["dtype"]))
    interp.invoke()
    print("[quantize] sanity inference OK — next: ./models/export_vela.sh", args.out)


if __name__ == "__main__":
    main()
