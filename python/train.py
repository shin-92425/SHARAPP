"""
Train the shape CNN and export it for the app.

    python train.py                  # synthetic shapes only
    python train.py --real dataset   # synthetic + the photos you collected

Outputs (copied straight into the Flutter project):
    ../assets/models/shape_model.tflite
    ../assets/models/labels.txt
"""
import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf

from shape_common import LABELS, SIL
from synth import make_dataset


def load_real(folder):
    xs, ys = [], []
    for idx, label in enumerate(LABELS):
        for path in glob.glob(os.path.join(folder, label, "*.png")):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (SIL, SIL):
                continue
            xs.append(img)
            ys.append(idx)
    if not xs:
        return np.zeros((0, SIL, SIL), np.uint8), np.zeros((0,), np.int64)
    return np.array(xs, np.uint8), np.array(ys, np.int64)


def jitter(x, rng):
    """Small random wobble so real photos are not memorised."""
    out = np.empty_like(x)
    for i, img in enumerate(x):
        a = rng.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((SIL / 2, SIL / 2), a, rng.uniform(0.92, 1.0))
        im = cv2.warpAffine(img, M, (SIL, SIL))
        k = rng.integers(0, 3)
        if k == 1:
            im = cv2.erode(im, np.ones((2, 2), np.uint8))
        elif k == 2:
            im = cv2.dilate(im, np.ones((2, 2), np.uint8))
        out[i] = im
    return out


def build_model(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input((SIL, SIL, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="folder made by collect_data.py")
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "models"))
    args = ap.parse_args()

    rng = np.random.default_rng(1)
    print("Rendering synthetic shapes ...")
    xs, ys = make_dataset(args.per_class, seed=0)
    xv, yv = make_dataset(max(50, args.per_class // 5), seed=999)

    if args.real:
        xr, yr = load_real(args.real)
        print("Real images found:", len(xr), np.bincount(yr, minlength=len(LABELS)))
        if len(xr):
            reps = 6
            xs = np.concatenate([xs] + [jitter(xr, rng) for _ in range(reps)] + [xr])
            ys = np.concatenate([ys] + [yr] * reps + [yr])
            # hold a slice of real photos out for an honest check
            sel = rng.random(len(xr)) < 0.2
            if sel.any():
                xv = np.concatenate([xv, xr[sel]])
                yv = np.concatenate([yv, yr[sel]])

    fx = (xs.astype(np.float32) / 255.0)[..., None]
    fv = (xv.astype(np.float32) / 255.0)[..., None]

    model = build_model(len(LABELS))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(fx, ys, validation_data=(fv, yv), epochs=args.epochs,
              batch_size=64, shuffle=True, verbose=2)

    loss, acc = model.evaluate(fv, yv, verbose=0)
    print("Validation accuracy: %.3f" % acc)
    pred = model.predict(fv, verbose=0).argmax(1)
    for i, l in enumerate(LABELS):
        m = yv == i
        print("  %-9s %.3f" % (l, (pred[m] == i).mean()))

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = conv.convert()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "shape_model.tflite"), "wb") as f:
        f.write(tflite)
    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        f.write("\n".join(LABELS) + "\n")
    print("Saved model (%d KB) to %s" % (len(tflite) // 1024, os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
