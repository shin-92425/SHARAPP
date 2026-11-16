"""
Draws the Shape Buddy launcher icon and writes it into the Android project,
replacing the default Flutter logo.

    python tools/make_icon.py                 # writes android/app/src/main/res/mipmap-*/ic_launcher.png
    python tools/make_icon.py --preview x.png # just a 512px preview
"""
import argparse
import math
import os

from PIL import Image, ImageDraw

CREAM = (255, 248, 238)
INK = (59, 63, 81)
BLUE = (79, 157, 166)

SIZES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}


def draw_icon(px=1024):
    s = px / 1024.0
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, px - 1, px - 1], radius=int(224 * s), fill=CREAM)

    # big blue circle
    d.ellipse([130 * s, 130 * s, 490 * s, 490 * s], fill=BLUE)
    # ink triangle
    d.polygon([(170 * s, 880 * s), (390 * s, 540 * s), (610 * s, 880 * s)], fill=INK)
    # blue star on the right
    cx, cy, ro, ri = 735 * s, 320 * s, 165 * s, 74 * s
    pts = []
    for i in range(10):
        r = ro if i % 2 == 0 else ri
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d.polygon(pts, fill=BLUE)
    # small square
    d.rounded_rectangle([690 * s, 640 * s, 890 * s, 840 * s], radius=int(28 * s), fill=BLUE)
    return img


def write_icons(res_dir):
    big = draw_icon(1024)
    for name, size in SIZES.items():
        folder = os.path.join(res_dir, "mipmap-" + name)
        os.makedirs(folder, exist_ok=True)
        big.resize((size, size), Image.LANCZOS).save(os.path.join(folder, "ic_launcher.png"))
    return list(SIZES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join("android", "app", "src", "main", "res"))
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()
    if a.preview:
        draw_icon(512).save(a.preview)
        print("preview written to", a.preview)
    else:
        write_icons(a.res)
        print("icons written under", a.res)
