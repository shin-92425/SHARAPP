"""
Synthetic training data  (version 2).

Each sample is rendered as a fake phone frame (480x720, portrait - the same
size as the app's upright preview) and pushed through the SAME
find_candidates()/silhouette() code the app uses, so the CNN trains on exactly
what it will see on the phone.

What the fake frames now contain:
  * shapes at many sizes (2.5% - 35% of the frame), rotated, stretched, in
    perspective, sometimes with rounded corners or drawn as an outline only
  * light or dark shapes, high AND low contrast
  * lighting gradients, hard shadows, a faint paper/table texture
  * defocus blur, motion blur, sensor noise
  * distractor blobs lying around (these become "other" samples when the app
    would have picked them instead of the real shape)

Class "other": shapes that are NOT one of the five (pentagon, hexagon, cross,
arrow, blobs ...) plus any distractor contour. It teaches the CNN to say "not
one of mine" instead of forcing a guess.
"""
import math

import cv2
import numpy as np

from shape_common import LABELS, CLASSES, OTHER, analyse_all

FRAME_W, FRAME_H = 480, 720

_XX, _YY = np.meshgrid(np.arange(FRAME_W, dtype=np.float32),
                       np.arange(FRAME_H, dtype=np.float32))
_GX = np.linspace(-1, 1, FRAME_W, dtype=np.float32)[None, :]
_GY = np.linspace(-1, 1, FRAME_H, dtype=np.float32)[:, None]


# ---------------------------------------------------------------- shape makers
# Each returns Nx2 points roughly inside the unit circle.

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


def _star(rng, points=5, inner=None):
    inner = rng.uniform(0.38, 0.55) if inner is None else inner
    pts = []
    for i in range(points * 2):
        r = 1.0 if i % 2 == 0 else inner
        a = -math.pi / 2 + i * math.pi / points
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


def _ngon(n):
    def make(rng):
        base = rng.uniform(0, 2 * np.pi)
        t = base + np.arange(n) * 2 * np.pi / n
        return np.stack([np.cos(t), np.sin(t)], 1)
    return make


def _rect(rng):
    a = rng.uniform(1.8, 3.2)
    return np.array([[-a, -1], [a, -1], [a, 1], [-a, 1]], np.float64) / a


def _ellipse(rng):
    a = rng.uniform(1.8, 2.8)
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    return np.stack([np.cos(t), np.sin(t) / a], 1)


def _cross(rng):
    w = rng.uniform(0.28, 0.5)
    return np.array([[-w, -1], [w, -1], [w, -w], [1, -w], [1, w], [w, w],
                     [w, 1], [-w, 1], [-w, w], [-1, w], [-1, -w], [-w, -w]], np.float64)


def _arrow(rng):
    s = rng.uniform(0.3, 0.5)
    return np.array([[-1, -s], [0.1, -s], [0.1, -1], [1, 0], [0.1, 1], [0.1, s], [-1, s]], np.float64)


def _lshape(rng):
    w = rng.uniform(0.35, 0.55)
    return np.array([[-1, -1], [-1 + 2 * w, -1], [-1 + 2 * w, 1 - 2 * w],
                     [1, 1 - 2 * w], [1, 1], [-1, 1]], np.float64)


def _semicircle(rng):
    t = np.linspace(0, np.pi, 60)
    pts = np.stack([np.cos(t), -np.sin(t)], 1)
    return pts + [0, 0.5]


def _blob(rng):
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    r = np.ones_like(t)
    for k in (2, 3, 4, 5):
        r += rng.uniform(0.05, 0.16) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    r = r / r.max()
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def _diamond(rng):
    a = rng.uniform(1.9, 2.8)
    return np.array([[0, -1], [1 / a * 1.0, 0], [0, 1], [-1 / a * 1.0, 0]], np.float64)


def _trapezoid(rng):
    k = rng.uniform(0.3, 0.6)
    return np.array([[-k, -1], [k, -1], [1, 1], [-1, 1]], np.float64)


OTHER_MAKERS = {
    "pentagon": _ngon(5), "hexagon": _ngon(6), "rect": _rect, "ellipse": _ellipse,
    "cross": _cross, "arrow": _arrow, "lshape": _lshape, "semicircle": _semicircle,
    "blob": _blob, "diamond": _diamond, "star4": lambda r: _star(r, 4),
    "star6": lambda r: _star(r, 6), "trapezoid": _trapezoid,
}
_OTHER_NAMES = sorted(OTHER_MAKERS)


# ------------------------------------------------------------------- rendering

def _random_poly(rng, kind):
    """kind: a class label or OTHER.  Returns polygon points in frame pixels."""
    if kind == OTHER:
        pts = OTHER_MAKERS[_OTHER_NAMES[rng.integers(len(_OTHER_NAMES))]](rng)
    else:
        pts = MAKERS[kind](rng)

    pts = pts * [rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)]     # stretch
    a = rng.uniform(0, 2 * np.pi)                                       # rotate
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = pts @ R.T

    # size: a target share of the frame, log-uniform 2.5% .. 35%
    frac = math.exp(rng.uniform(math.log(0.025), math.log(0.35)))
    x, y = pts[:, 0], pts[:, 1]
    unit_area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
    s = math.sqrt(frac * FRAME_W * FRAME_H / max(unit_area, 1e-6))
    e = np.abs(pts).max(axis=0)
    s = min(s, (FRAME_W / 2 - 14) / e[0], (FRAME_H / 2 - 14) / e[1])
    ext = np.abs(pts * s).max(axis=0)
    cx = FRAME_W / 2 + rng.uniform(-1, 1) * max(0.0, FRAME_W / 2 - ext[0] - 12) * 0.85
    cy = FRAME_H / 2 + rng.uniform(-1, 1) * max(0.0, FRAME_H / 2 - ext[1] - 12) * 0.85
    pts = pts * s + [cx, cy]

    # perspective: phone held at an angle
    src = np.float32([[0, 0], [FRAME_W, 0], [FRAME_W, FRAME_H], [0, FRAME_H]])
    amt = rng.choice([0.0, 0.03, 0.06, 0.09])
    j = (rng.uniform(-1, 1, (4, 2)) * [FRAME_W * amt, FRAME_H * amt]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, src + j)
    pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    return pts


def _ellipse_kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def render_frame(kind, rng, distractors=None):
    """
    kind: one of LABELS or OTHER.
    Returns (frame uint8 HxW, gt mask uint8 HxW for the intended shape).
    """
    pts = _random_poly(rng, kind)
    poly = np.round(pts).astype(np.int32)
    mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    ys, xs = np.nonzero(mask)
    size_px = max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    if rng.random() < 0.25:                      # rounded corners (foam / cut-outs)
        k = int(rng.integers(3, 4 + max(1, size_px // 14)))
        k = min(k, 9) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse_kernel(k))
    gt = (mask > 0).astype(np.uint8)

    # tones
    dark_shape = rng.random() < 0.5
    if rng.random() < 0.7:                       # strong contrast
        if dark_shape:
            bg, fg = int(rng.integers(150, 250)), int(rng.integers(10, 110))
        else:
            bg, fg = int(rng.integers(10, 100)), int(rng.integers(160, 250))
    else:                                        # weak contrast
        d = int(rng.integers(22, 60))
        bg = int(rng.integers(110, 200))
        fg = bg - d if dark_shape else bg + d

    frame = np.full((FRAME_H, FRAME_W), float(bg), np.float32)
    g = rng.uniform(0, 45)
    frame += _GX * rng.uniform(-g, g) + _GY * rng.uniform(-g, g)       # lighting ramp
    if rng.random() < 0.4:                                              # paper texture
        tex = rng.normal(0, 1, (int(rng.integers(8, 20)), int(rng.integers(12, 30)))).astype(np.float32)
        tex = cv2.resize(tex, (FRAME_W, FRAME_H), interpolation=cv2.INTER_CUBIC)
        frame += tex * rng.uniform(2, 9)
    img = np.clip(frame, 0, 255).astype(np.uint8)

    # distractors lying around (drawn first, the real shape goes on top)
    n_d = 0 if distractors is None else distractors
    for _ in range(n_d):
        kk = int(rng.integers(3, 8))
        c = rng.uniform([0, 0], [FRAME_W, FRAME_H])
        r = rng.uniform(35, 160)
        ang = np.sort(rng.uniform(0, 2 * np.pi, kk))
        rr = r * rng.uniform(0.6, 1.0, kk)
        dp = np.round(np.stack([c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang)], 1)).astype(np.int32)
        tone = int(np.clip(fg + rng.choice([-1, 1]) * rng.integers(12, 70), 0, 255))
        cv2.fillPoly(img, [dp], tone)

    if rng.random() < 0.15:                      # outline only (drawn / printed)
        t = int(rng.integers(4, 10))
        inner = cv2.erode(mask, _ellipse_kernel(2 * t + 1))
        ring = (mask > 0) & (inner == 0)
        img[(mask > 0) & ~ring] = bg
        img[ring] = fg
    else:
        img[mask > 0] = fg

    if rng.random() < 0.3:                       # hard-edged shadow (hand / phone)
        if rng.random() < 0.5:
            p0 = rng.uniform([0, 0], [FRAME_W, FRAME_H])
            ang = rng.uniform(0, 2 * np.pi)
            d = np.array([math.cos(ang), math.sin(ang)], np.float32)
            sh = (((_XX - p0[0]) * d[0] + (_YY - p0[1]) * d[1]) > 0).astype(np.float32)
        else:
            p0 = rng.uniform([0, 0], [FRAME_W, FRAME_H])
            ax, ay = rng.uniform(90, 260), rng.uniform(90, 260)
            sh = ((((_XX - p0[0]) / ax) ** 2 + ((_YY - p0[1]) / ay) ** 2) < 1).astype(np.float32)
        sh = cv2.GaussianBlur(sh, (0, 0), float(rng.uniform(2, 12)))
        img = (img.astype(np.float32) * (1 - rng.uniform(0.25, 0.5) * sh)).astype(np.uint8)

    r = rng.random()                             # camera blur
    if r < 0.25:                                 # motion blur
        k = int(rng.integers(5, 16))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D(((k - 1) / 2.0, (k - 1) / 2.0), rng.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, M, (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    elif r < 0.85:                               # defocus
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    noise = rng.standard_normal(img.shape, dtype=np.float32) * np.float32(rng.uniform(1, 12))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, gt


def _iou(contour, size_wh, gt):
    sw, sh = size_wh
    pts = contour.copy()
    pts[:, 0] *= FRAME_W / sw
    pts[:, 1] *= FRAME_H / sh
    m = np.zeros((FRAME_H, FRAME_W), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    inter = int((m & gt).sum())
    union = int((m | gt).sum())
    return inter / union if union else 0.0


def make_dataset(per_class, seed=0, return_meta=False):
    """
    Returns (silhouettes N x 64 x 64 uint8, labels N int64) for CLASSES
    (the five shapes + "other"), `per_class` of each.  With return_meta=True
    also returns a list saying where each sample came from.
    """
    rng = np.random.default_rng(seed)
    xs, ys, meta = [], [], []
    extras = []          # distractor contours the app would have shown the CNN

    def one(kind, want_extras):
        n_d = int(rng.choice([0, 0, 1, 2, 3]))
        frame, gt = render_frame(kind, rng, distractors=n_d)
        cands, size = analyse_all(frame)
        match, others = None, []
        for sil, contour in cands:
            v = _iou(contour, size, gt)
            if v >= 0.8 and match is None:
                match = sil
            elif v < 0.25:
                others.append(sil)
        if want_extras:
            extras.extend(others[:2])
        return match

    for idx, label in enumerate(LABELS):
        made = 0
        while made < per_class:
            sil = one(label, True)
            if sil is None:
                continue
            xs.append(sil); ys.append(idx); meta.append("synth-" + label)
            made += 1

    o_idx = CLASSES.index(OTHER)
    n_direct = int(per_class * 0.6)
    made = 0
    while made < n_direct:
        sil = one(OTHER, True)
        if sil is None:
            continue
        xs.append(sil); ys.append(o_idx); meta.append("synth-other-shape")
        made += 1
    take = min(len(extras), per_class - n_direct)
    for sil in extras[:take]:
        xs.append(sil); ys.append(o_idx); meta.append("synth-other-distractor")
        made += 1
    while made < per_class:      # not enough distractor samples: more odd shapes
        sil = one(OTHER, False)
        if sil is None:
            continue
        xs.append(sil); ys.append(o_idx); meta.append("synth-other-shape")
        made += 1

    xs = np.array(xs, np.uint8)
    ys = np.array(ys, np.int64)
    return (xs, ys, meta) if return_meta else (xs, ys)


if __name__ == "__main__":
    x, y, m = make_dataset(20, return_meta=True)
    print(x.shape, y.shape, np.bincount(y))
    from collections import Counter
    print(Counter(m))
