"""
Contour-finding benchmark (no CNN needed).

Renders phone-sized portrait frames (480x720, like the app's upright preview)
of one shape under different difficulties, runs a pipeline's find_shape(), and
scores the contour it picked against the true shape (IoU).

    python bench.py                  # score shape_common.py
    python bench.py --module old_common
"""
import argparse
import importlib
import math

import cv2
import numpy as np

from synth import MAKERS

W, H = 480, 720
LABELS = ["circle", "heart", "square", "star", "triangle"]
CONDITIONS = ["easy", "gradient", "shadow", "lowcontrast", "clutter",
              "small", "blurnoise", "outline"]


def shape_points(label, rng, area_frac):
    """Shape polygon scaled so it covers ~area_frac of the frame."""
    pts = MAKERS[label](rng)
    pts = pts * [rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)]
    a = rng.uniform(0, 2 * np.pi)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = pts @ R.T
    x, y = pts[:, 0], pts[:, 1]
    unit_area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
    s = math.sqrt(area_frac * W * H / unit_area)
    e = np.abs(pts).max(axis=0)
    s = min(s, (W / 2 - 14) / e[0], (H / 2 - 14) / e[1])   # always fits in frame
    ext = np.abs(pts * s).max(axis=0)          # half-extent in px
    cx = W / 2 + rng.uniform(-1, 1) * max(0, W / 2 - ext[0] - 12) * 0.8
    cy = H / 2 + rng.uniform(-1, 1) * max(0, H / 2 - ext[1] - 12) * 0.8
    return pts * s + [cx, cy]


def render(label, cond, rng):
    area_frac = rng.uniform(0.025, 0.05) if cond == "small" else rng.uniform(0.06, 0.28)
    pts = shape_points(label, rng, area_frac)

    dark_on_light = rng.random() < 0.5
    if dark_on_light:
        bg, fg = rng.integers(170, 250), rng.integers(10, 110)
    else:
        bg, fg = rng.integers(10, 90), rng.integers(160, 250)
    if cond == "lowcontrast":
        d = rng.integers(22, 42)
        bg = rng.integers(110, 200)
        fg = bg - d if dark_on_light else bg + d

    frame = np.full((H, W), float(bg), np.float32)
    gmax = 60 if cond == "gradient" else 15
    frame += (np.linspace(-1, 1, W)[None, :] * rng.uniform(-gmax, gmax)
              + np.linspace(-1, 1, H)[:, None] * rng.uniform(-gmax, gmax))
    img = np.clip(frame, 0, 255).astype(np.uint8)

    if cond == "clutter":
        # distractors in the same tone as the shape: one big blob, a few small
        # ones, and a paper-edge-like band; the real shape is drawn on top.
        for _ in range(rng.integers(2, 5)):
            k = rng.integers(3, 7)
            c = rng.uniform([0, 0], [W, H])
            r = rng.uniform(40, 150)
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            rr = r * rng.uniform(0.6, 1.0, k)
            poly = np.stack([c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang)], 1)
            cv2.fillPoly(img, [np.round(poly).astype(np.int32)],
                         int(fg + rng.integers(-25, 25)))
        x = int(rng.integers(0, W // 3))
        cv2.rectangle(img, (0, 0), (x, H), int(fg + rng.integers(-15, 15)), -1)

    poly = np.round(pts).astype(np.int32)
    gt = np.zeros((H, W), np.uint8)
    cv2.fillPoly(gt, [poly], 1)
    if cond == "outline":
        cv2.fillPoly(img, [poly], int(bg))
        cv2.polylines(img, [poly], True, int(fg), int(rng.integers(4, 9)))
    else:
        cv2.fillPoly(img, [poly], int(fg))

    if cond == "shadow":
        # hard-edged shadow across a random part of the frame
        sh = np.zeros((H, W), np.uint8)
        p0 = rng.uniform([0, 0], [W, H])
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([math.cos(ang), math.sin(ang)])
        yy, xx = np.mgrid[0:H, 0:W]
        sh[((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) > 0] = 1
        sh = cv2.GaussianBlur(sh.astype(np.float32), (0, 0), 6)
        img = (img.astype(np.float32) * (1 - 0.45 * sh)).astype(np.uint8)

    if cond == "blurnoise":
        k = int(rng.integers(7, 16))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, M, (k, k))
        ker /= ker.sum()
        img = cv2.filter2D(img, -1, ker)
        sigma = rng.uniform(8, 15)
    else:
        if rng.random() < 0.7:
            kk = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (kk, kk), 0)
        sigma = rng.uniform(1, 8)
    img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return img, gt


def iou_of(contour, size_wh, gt):
    if contour is None:
        return 0.0
    sw, sh = size_wh
    pts = contour.copy()
    pts[:, 0] *= W / sw
    pts[:, 1] *= H / sh
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    inter = (m & gt).sum()
    union = (m | gt).sum()
    return inter / union if union else 0.0


def run(mod, n, seed):
    rng = np.random.default_rng(seed)
    rows = {}
    for cond in CONDITIONS:
        ok = wrong = none = 0
        ious = []
        for i in range(n):
            label = LABELS[i % len(LABELS)]
            img, gt = render(label, cond, rng)
            contour, size = mod.find_shape(img)
            v = iou_of(contour, size, gt)
            if contour is None:
                none += 1
            elif v >= 0.80:
                ok += 1
                ious.append(v)
            else:
                wrong += 1
        rows[cond] = (ok / n, wrong / n, none / n, float(np.mean(ious)) if ious else 0.0)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="shape_common")
    ap.add_argument("-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    mod = importlib.import_module(a.module)
    res = run(mod, a.n, a.seed)
    print(f"module: {a.module}   frames per condition: {a.n}")
    print(f"{'condition':<12}{'good':>8}{'wrong':>8}{'none':>8}{'meanIoU':>9}")
    tot = 0
    for c, (g, w, nn, m) in res.items():
        print(f"{c:<12}{g*100:7.1f}%{w*100:7.1f}%{nn*100:7.1f}%{m:9.3f}")
        tot += g
    print(f"{'AVERAGE':<12}{tot/len(res)*100:7.1f}%")


def run_topk(mod, n, seed, k=3):
    """Like run(), but 'good' = ANY of the top-k candidates matches the truth."""
    rng = np.random.default_rng(seed)
    rows = {}
    for cond in CONDITIONS:
        top1 = anyk = 0
        for i in range(n):
            label = LABELS[i % len(LABELS)]
            img, gt = render(label, cond, rng)
            cands, size = mod.find_candidates(img, k)
            ious = [iou_of(c, size, gt) for c in cands]
            if ious and ious[0] >= 0.8:
                top1 += 1
            if any(v >= 0.8 for v in ious):
                anyk += 1
        rows[cond] = (top1 / n, anyk / n)
    return rows
