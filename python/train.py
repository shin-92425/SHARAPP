"""
Train the shape CNN and export it for the app  (version 2).

    python train.py                  # synthetic shapes only
    python train.py --real dataset   # synthetic + the photos you collected

What changed compared with version 1
  * HONEST validation: real photos are split into train / held-out BEFORE any
    augmentation, as contiguous blocks of files (consecutive frames from the
    collector are near-identical, so a random split would leak). A few files
    around each cut are dropped as a safety gap.
  * Real silhouettes are augmented the way the app would really see them:
    rotate / stretch / perspective / flip on a roomy canvas, then crop and
    re-fit to the bounding box exactly like silhouette_points() does.
  * New class "other" (not one of the five shapes) so unknown shapes and
    distractors are rejected instead of forced into a class.
  * Separate reports for synthetic and real validation, with a confusion
    matrix and accuracy at the app's confidence threshold.
  * Early stopping keeps the best epoch.

Outputs (copied straight into the Flutter project):
    ../assets/models/shape_model.tflite
    ../assets/models/labels.txt          (6 lines: 5 shapes + other)
"""
import argparse
import glob
import os

import cv2
import numpy as np

from shape_common import CLASSES, LABELS, SIL, SIL_PAD
from synth import make_dataset

MIN_CONFIDENCE = 0.80      # must match minConfidence in shape_detector.dart
FIT = SIL - 2 * SIL_PAD    # long side of a normalised silhouette (52 px)
CANVAS = 192               # roomy canvas so rotated corners never get clipped


# ----------------------------------------------------------------- real photos

def load_real(folder, hold_out=0.2, gap=3):
    """
    Reads dataset/<class>/*.png (64x64 silhouettes written by collect_data.py;
    an optional dataset/other/ folder works too).

    Returns (x_train, y_train, x_val, y_val). Files are sorted by name and the
    LAST `hold_out` share of each class becomes the validation block; `gap`
    files before the cut are dropped so neighbours of the cut do not leak.
    """
    xt, yt, xv, yv = [], [], [], []
    for idx, label in enumerate(CLASSES):
        paths = sorted(glob.glob(os.path.join(folder, label, "*.png")))
        imgs = []
        for p in paths:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is not None and im.shape == (SIL, SIL):
                imgs.append(im)
        n = len(imgs)
        if n == 0:
            continue
        n_val = int(round(n * hold_out)) if n >= 10 else 0
        cut = n - n_val
        train = imgs[:max(0, cut - (gap if n_val else 0))]
        val = imgs[cut:]
        xt += train; yt += [idx] * len(train)
        xv += val;   yv += [idx] * len(val)

    def pack(xs, ys):
        if not xs:
            return np.zeros((0, SIL, SIL), np.uint8), np.zeros((0,), np.int64)
        return np.array(xs, np.uint8), np.array(ys, np.int64)

    return (*pack(xt, yt), *pack(xv, yv))


def _refit(mask):
    """Crop to the bounding box and re-fit to FIT px, centred (like the app)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    s = FIT / max(crop.shape)
    nw, nh = max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    crop = (crop > 127).astype(np.uint8) * 255
    out = np.zeros((SIL, SIL), np.uint8)
    ox, oy = (SIL - nw) // 2, (SIL - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = crop
    return out


def augment_one(img, rng):
    """One realistic variation of a 64x64 silhouette (None if it degenerates)."""
    big = np.zeros((CANVAS, CANVAS), np.uint8)
    up = cv2.resize(img, (SIL * 2, SIL * 2), interpolation=cv2.INTER_LINEAR)
    o = (CANVAS - SIL * 2) // 2
    big[o:o + SIL * 2, o:o + SIL * 2] = up

    c = CANVAS / 2.0
    a = rng.uniform(0, 360)
    sx, sy = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((c, c), a, 1.0)
    M = np.vstack([M, [0, 0, 1]])
    S = np.array([[sx, 0, c * (1 - sx)], [0, sy, c * (1 - sy)], [0, 0, 1]])
    H = S @ M
    if rng.random() < 0.7:                           # perspective wobble
        amt = rng.choice([0.03, 0.06, 0.09]) * CANVAS
        src = np.float32([[0, 0], [CANVAS, 0], [CANVAS, CANVAS], [0, CANVAS]])
        P = cv2.getPerspectiveTransform(src, src + rng.uniform(-amt, amt, (4, 2)).astype(np.float32))
        H = P @ H
    if rng.random() < 0.5:                           # mirror (all five shapes are symmetric)
        F = np.array([[-1, 0, CANVAS], [0, 1, 0], [0, 0, 1]], np.float64)
        H = F @ H
    warped = cv2.warpPerspective(big, H, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    mask = (warped > 127).astype(np.uint8) * 255

    k = rng.integers(0, 4)                           # rough / bloated edges
    if k == 1:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    elif k == 2:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return _refit(mask)


def augment(x, y, reps, rng):
    xs, ys = [], []
    for _ in range(reps):
        for im, lab in zip(x, y):
            out = augment_one(im, rng)
            if out is not None:
                xs.append(out); ys.append(lab)
    if not xs:
        return np.zeros((0, SIL, SIL), np.uint8), np.zeros((0,), np.int64)
    return np.array(xs, np.uint8), np.array(ys, np.int64)


# ----------------------------------------------------------------------- model

def build_model(n_classes):
    import tensorflow as tf
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


# --------------------------------------------------------------------- reports

def report(name, probs, y):
    """Accuracy, per-class recall, confusion matrix, accuracy at the app's threshold."""
    if len(y) == 0:
        print("\n[%s] no samples" % name)
        return
    pred = probs.argmax(1)
    conf = probs.max(1)
    n = len(CLASSES)
    print("\n[%s]  %d samples   accuracy %.3f" % (name, len(y), (pred == y).mean()))
    sure = conf >= MIN_CONFIDENCE
    if sure.any():
        print("  at confidence >= %.2f: answers %.0f%% of samples, %.3f of those correct"
              % (MIN_CONFIDENCE, 100 * sure.mean(), (pred[sure] == y[sure]).mean()))
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print("  rows = true class, columns = predicted")
    print("  %-9s" % "" + "".join("%9s" % c[:8] for c in CLASSES) + "   recall")
    for i, c in enumerate(CLASSES):
        tot = cm[i].sum()
        rec = cm[i, i] / tot if tot else float("nan")
        print("  %-9s" % c + "".join("%9d" % v for v in cm[i]) + "   %.3f" % rec)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="folder made by collect_data.py")
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--reps", type=int, default=8, help="augmented copies of each real photo")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "models"))
    args = ap.parse_args()

    import tensorflow as tf

    rng = np.random.default_rng(1)
    print("Rendering synthetic shapes (this takes a few minutes) ...")
    xs, ys = make_dataset(args.per_class, seed=0)
    xv_s, yv_s = make_dataset(max(60, args.per_class // 5), seed=999)

    xr_v, yr_v = np.zeros((0, SIL, SIL), np.uint8), np.zeros((0,), np.int64)
    if args.real:
        xr_t, yr_t, xr_v, yr_v = load_real(args.real)
        print("Real images  train:", len(xr_t), np.bincount(yr_t, minlength=len(CLASSES)),
              " held-out:", len(xr_v), np.bincount(yr_v, minlength=len(CLASSES)))
        if len(xr_t):
            xa, ya = augment(xr_t, yr_t, args.reps, rng)
            xs = np.concatenate([xs, xa, xr_t])
            ys = np.concatenate([ys, ya, yr_t])
        if len(xr_v) == 0:
            print("WARNING: too few real photos per class to hold any out; "
                  "only the synthetic validation is meaningful.")

    xv = np.concatenate([xv_s, xr_v]) if len(xr_v) else xv_s
    yv = np.concatenate([yv_s, yr_v]) if len(xr_v) else yv_s

    f = lambda a: (a.astype(np.float32) / 255.0)[..., None]
    fx, fv = f(xs), f(xv)

    model = build_model(len(CLASSES))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                            restore_best_weights=True)
    model.fit(fx, ys, validation_data=(fv, yv), epochs=args.epochs,
              batch_size=64, shuffle=True, verbose=2, callbacks=[stop])

    report("synthetic validation", model.predict(f(xv_s), verbose=0), yv_s)
    if len(xr_v):
        report("REAL held-out photos (the number that matters)",
               model.predict(f(xr_v), verbose=0), yr_v)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = conv.convert()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "shape_model.tflite"), "wb") as fh:
        fh.write(tflite)
    with open(os.path.join(args.out, "labels.txt"), "w") as fh:
        fh.write("\n".join(CLASSES) + "\n")
    print("\nSaved model (%d KB) to %s" % (len(tflite) // 1024, os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
