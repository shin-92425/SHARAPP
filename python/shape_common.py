"""
Shared shape-finding pipeline  (version 2).

IMPORTANT: lib/services/shape_detector.dart is a step-by-step mirror of this
file. If you change a constant or a step here, change it there too, then
retrain the model (python train.py --real dataset).

Pipeline (works on a single upright grayscale frame):
  1. shrink so the long side is MAX_SIDE pixels (INTER_AREA = proper averaging)
  2. 3x3 Gaussian blur
  3. flatten the lighting: divide by a heavily blurred copy of the picture, so
     gradients and shadows stop fooling the threshold
  4. Otsu threshold, tried both ways (dark-on-light and light-on-dark)
  5. external contours -> keep sensible ones: right size, not touching the
     frame border, not a thin sliver, not too "spiky/hollow"
  6. rank them (big and near the centre) and keep the best few (TOP_K)
  7. each candidate becomes a 64x64 filled silhouette for the CNN; the app
     asks the CNN about every candidate and keeps the most convincing one

Public API kept from version 1:  find_shape, silhouette, silhouette_points,
analyse, LABELS, SIL, MAX_SIDE ...   New:  find_candidates, CLASSES, OTHER.
"""
import cv2
import numpy as np

MAX_SIDE = 200          # long side of the working image
BORDER_MARGIN = 2       # contour may not come this close to the image edge
MIN_AREA_FRAC = 0.02    # smallest shape (fraction of the frame)
MAX_AREA_FRAC = 0.60    # biggest shape (fraction of the frame)
SIL = 64                # silhouette size fed to the CNN
SIL_PAD = 6             # empty margin inside the silhouette

FLATTEN_DIV = 8.0       # lighting-flatten blur sigma = long side / FLATTEN_DIV
MIN_SOLIDITY = 0.35     # contour area / convex-hull area (a star is ~0.47)
MAX_ELONGATION = 4.0    # long side / short side of the tightest box
TOP_K = 3               # candidates handed to the CNN

LABELS = ["circle", "heart", "square", "star", "triangle"]
OTHER = "other"
CLASSES = LABELS + [OTHER]      # what the CNN is trained to tell apart


def shrink(gray):
    """Area-averaging shrink so the long side is MAX_SIDE."""
    h, w = gray.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1.0:
        return gray
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def flatten(blur):
    """Divide out the slowly-changing lighting (shadows, gradients)."""
    h, w = blur.shape[:2]
    bg = cv2.GaussianBlur(blur, (0, 0), max(h, w) / FLATTEN_DIV)
    flat = blur.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 128.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def polygon_area(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def _quality_ok(pts, area):
    """Reject slivers (paper edges, table seams) and wispy/hollow blobs."""
    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0 or area / hull_area < MIN_SOLIDITY:
        return False
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts.astype(np.float32).reshape(-1, 1, 2))
    short, long_ = min(rw, rh), max(rw, rh)
    if short < 1.0 or long_ / short > MAX_ELONGATION:
        return False
    return True


def _bbox_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def find_candidates(gray, k=TOP_K):
    """
    Returns ([contour Nx2 float arrays, best first], (w, h) of the shrunken
    image).  Contour points are in the shrunken image's pixels.
    """
    small = shrink(gray)
    h, w = small.shape[:2]
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    flat = flatten(blur)
    frame_area = float(w * h)
    cx, cy = w / 2.0, h / 2.0
    half_diag = (cx * cx + cy * cy) ** 0.5

    found = []   # (score, bbox, contour)
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(flat, 0, 255, mode | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) < 5:
                continue
            area = polygon_area(pts)
            if area < MIN_AREA_FRAC * frame_area or area > MAX_AREA_FRAC * frame_area:
                continue
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            if (x0 <= BORDER_MARGIN or y0 <= BORDER_MARGIN or
                    x1 >= w - 1 - BORDER_MARGIN or y1 >= h - 1 - BORDER_MARGIN):
                continue
            if not _quality_ok(pts, area):
                continue
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dist = (((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5) / half_diag
            score = area * (1.0 - 0.5 * dist)
            found.append((score, (float(x0), float(y0), float(x1), float(y1)),
                          pts.astype(np.float64)))

    found.sort(key=lambda t: -t[0])
    kept = []
    for score, box, pts in found:
        if any(_bbox_iou(box, kb) > 0.85 for _, kb, _ in kept):
            continue    # same blob seen by both thresholds
        kept.append((score, box, pts))
        if len(kept) >= k:
            break
    return [p for _, _, p in kept], (w, h)


def find_shape(gray):
    """
    Single best candidate (the old API).  Returns (contour Nx2 float array in
    the shrunken image's pixels, (width, height)) or (None, size).
    """
    cands, size = find_candidates(gray, k=1)
    return (cands[0] if cands else None), size


def silhouette_points(contour):
    """Contour (Nx2) -> the same points mapped into the SILxSIL square."""
    x0, y0 = contour.min(axis=0)
    x1, y1 = contour.max(axis=0)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    s = (SIL - 2 * SIL_PAD) / max(bw, bh)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = np.empty_like(contour)
    out[:, 0] = (contour[:, 0] - cx) * s + SIL / 2.0
    out[:, 1] = (contour[:, 1] - cy) * s + SIL / 2.0
    return out


def silhouette(contour):
    """
    64x64 uint8 image (255 inside the shape, 0 outside).
    Even-odd scanline fill sampled at pixel centres; the Dart app uses the
    identical algorithm so both sides produce the same picture.
    """
    pts = silhouette_points(contour)
    img = np.zeros((SIL, SIL), np.uint8)
    nxt = np.roll(pts, -1, axis=0)
    x0, y0, x1, y1 = pts[:, 0], pts[:, 1], nxt[:, 0], nxt[:, 1]
    for y in range(SIL):
        sy = y + 0.5
        m = (y0 <= sy) != (y1 <= sy)
        if not m.any():
            continue
        xs = x0[m] + (sy - y0[m]) * (x1[m] - x0[m]) / (y1[m] - y0[m])
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a = max(0, int(np.ceil(xs[k] - 0.5)))
            b = min(SIL, int(np.ceil(xs[k + 1] - 0.5)))
            if b > a:
                img[y, a:b] = 255
    return img


def analyse(gray):
    """Convenience: gray frame -> (silhouette or None, contour or None, size)."""
    contour, size = find_shape(gray)
    if contour is None:
        return None, None, size
    return silhouette(contour), contour, size


def analyse_all(gray, k=TOP_K):
    """Gray frame -> ([(silhouette, contour), ...] best first, size)."""
    cands, size = find_candidates(gray, k)
    return [(silhouette(c), c) for c in cands], size
