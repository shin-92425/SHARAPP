"""
Synthetic training data.

Each sample is rendered as a fake camera frame (a shape on a background, with
random rotation, perspective, brightness, blur and noise) and then pushed
through the SAME find_shape()/silhouette() code the app uses. That way the
CNN trains on exactly what it will see on the phone.
"""
import math
import cv2
import numpy as np
from shape_common import LABELS, analyse

FRAME_W, FRAME_H = 320, 240


def _circle(rng):
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], 1)


def _square(rng):
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)


def _triangle(rng):
    pts = []
    base = rng.uniform(0, 2 * np.pi)
    for i in range(3):
        a = base + i * 2 * np.pi / 3 + rng.uniform(-0.25, 0.25)
        r = rng.uniform(0.85, 1.0)
        pts.append([r * math.cos(a), r * math.sin(a)])
    return np.array(pts)


def _star(rng):
    inner = rng.uniform(0.38, 0.55)
    pts = []
    for i in range(10):
        r = 1.0 if i % 2 == 0 else inner
        a = -math.pi / 2 + i * math.pi / 5
        pts.append([r * math.cos(a), r * math.sin(a)])
    return np.array(pts)


def _heart(rng):
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], 1) / 16.0
    pts[:, 1] += 0.05
    return pts


MAKERS = {"circle": _circle, "heart": _heart, "square": _square,
          "star": _star, "triangle": _triangle}


def render_frame(label, rng):
    pts = MAKERS[label](rng)

    # a little stretch so shapes aren't perfectly regular (viewing angle)
    sx, sy = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    pts = pts * [sx, sy]

    # random rotation
    a = rng.uniform(0, 2 * np.pi)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = pts @ R.T

    # random size / position
    size = rng.uniform(50, 100)
    cx = FRAME_W / 2 + rng.uniform(-40, 40)
    cy = FRAME_H / 2 + rng.uniform(-30, 30)
    pts = pts * size + [cx, cy]

    # mild perspective
    src = np.float32([[0, 0], [FRAME_W, 0], [FRAME_W, FRAME_H], [0, FRAME_H]])
    j = rng.uniform(-18, 18, (4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, src + j)
    pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)

    # background / shape tones with guaranteed contrast
    if rng.random() < 0.5:
        bg, fg = rng.integers(170, 250), rng.integers(10, 110)
    else:
        bg, fg = rng.integers(10, 90), rng.integers(160, 250)
    frame = np.full((FRAME_H, FRAME_W), bg, np.float32)

    # soft lighting gradient
    gx = np.linspace(-1, 1, FRAME_W)[None, :] * rng.uniform(-18, 18)
    gy = np.linspace(-1, 1, FRAME_H)[:, None] * rng.uniform(-18, 18)
    frame += gx + gy

    img = frame.astype(np.uint8)
    poly = np.round(pts).astype(np.int32)
    if rng.random() < 0.5:
        cv2.fillPoly(img, [poly], int(fg))
    else:  # hand-drawn style outline, filled inside with a similar tone
        cv2.fillPoly(img, [poly], int(fg))
        k = rng.integers(0, 3)
        if k:
            img = cv2.erode(img, np.ones((k, k), np.uint8)) if fg > bg else img

    if rng.random() < 0.7:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    noise = rng.normal(0, rng.uniform(1, 8), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def make_dataset(per_class, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for idx, label in enumerate(LABELS):
        made = 0
        while made < per_class:
            sil, _, _ = analyse(render_frame(label, rng))
            if sil is None:
                continue
            xs.append(sil)
            ys.append(idx)
            made += 1
    xs = np.array(xs, np.uint8)
    ys = np.array(ys, np.int64)
    return xs, ys


if __name__ == "__main__":
    x, y = make_dataset(20)
    print(x.shape, y.shape, np.bincount(y))
