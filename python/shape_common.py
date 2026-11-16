"""
Shared shape-finding pipeline.

IMPORTANT: lib/services/shape_detector.dart is a line-by-line mirror of this
file. If you change a constant or a step here, change it there too, then
retrain the model.

Pipeline (works on a single grayscale frame):
  1. shrink so the long side is MAX_SIDE pixels (nearest neighbour)
  2. 3x3 box blur
  3. Otsu threshold, tried both ways (dark-on-light and light-on-dark)
  4. external contours -> keep the ones that are a sensible size and do not
     touch the frame border
  5. pick the best one (big and near the centre)
  6. turn that contour into a 64x64 filled silhouette for the CNN
"""
import cv2
import numpy as np

MAX_SIDE = 200          # long side of the working image
BORDER_MARGIN = 2       # contour may not come this close to the image edge
MIN_AREA_FRAC = 0.02    # smallest shape (fraction of the frame)
MAX_AREA_FRAC = 0.60    # biggest shape (fraction of the frame)
SIL = 64                # silhouette size fed to the CNN
SIL_PAD = 6             # empty margin inside the silhouette

LABELS = ["circle", "heart", "square", "star", "triangle"]


def shrink(gray):
    """Nearest-neighbour shrink so the long side is MAX_SIDE."""
    h, w = gray.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1.0:
        return gray
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_NEAREST)


def polygon_area(pts):
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def find_shape(gray):
    """
    Returns (contour Nx2 float array in the shrunken image's pixels,
             (width, height) of the shrunken image) or (None, size).
    """
    small = shrink(gray)
    h, w = small.shape[:2]
    blur = cv2.blur(small, (3, 3))
    frame_area = float(w * h)
    cx, cy = w / 2.0, h / 2.0
    half_diag = (cx * cx + cy * cy) ** 0.5

    best, best_score = None, 0.0
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(blur, 0, 255, mode | cv2.THRESH_OTSU)
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
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dist = (((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5) / half_diag
            score = area * (1.0 - 0.5 * dist)
            if score > best_score:
                best_score, best = score, pts.astype(np.float64)
    return best, (w, h)


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
