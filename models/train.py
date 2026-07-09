"""Transfer-learning training for the defect classifier.

Trains a MobileNetV2-based binary classifier (good / defect) on an image
folder dataset, then exports a float SavedModel for quantization.

Dataset layout (MVTec-AD style, or your own photos):
    data/
      train/good/*.jpg      train/defect/*.jpg
      val/good/*.jpg        val/defect/*.jpg

Usage:
    pip install -r models/requirements-train.txt
    python models/train.py --data data --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # feature extraction; fine-tune later if needed
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def load_datasets(data_dir: str, batch: int):
    common = dict(image_size=IMG_SIZE, batch_size=batch, label_mode="categorical",
                  class_names=["good", "defect"])
    train = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/train", shuffle=True, **common)
    val = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/val", shuffle=False, **common)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomBrightness(0.1),
    ])
    train = train.map(lambda x, y: (aug(x, training=True), y),
                      num_parallel_calls=tf.data.AUTOTUNE)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="models/saved_model")
    args = parser.parse_args()

    train, val = load_datasets(args.data, args.batch)
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs)

    loss, acc = model.evaluate(val)
    print(f"[eval] val_accuracy={acc:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.export(args.out)
    print(f"[export] SavedModel written to {args.out}")
    print("Next: python models/quantize.py --data data --saved-model", args.out)


if __name__ == "__main__":
    main()
