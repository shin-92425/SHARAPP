"""
Collect real training images through your phone camera (DroidCam).

Setup
  1. Install DroidCam on the phone and the DroidCam client on the laptop
     (or use the Wi-Fi mode and note the address it shows, e.g. 192.168.1.5).
  2. Put paper shapes (dark on white works best) in front of the phone.
  3. Run, for example:
        python collect_data.py --source http://192.168.1.5:4747/video
     or, for the USB / DroidCam-client virtual webcam:
        python collect_data.py --source 1       (try 0, 1, 2 ... until you see the phone)

Keys (click the video window first)
  1 2 3 4 5   choose the shape you are showing:
              1 circle  2 heart  3 square  4 star  5 triangle
  SPACE       single capture
  C           continuous capture on/off (saves ~3 per second while a shape is seen)
  P           pause / resume the live view
  Q or ESC    exit

Saved to  dataset/<shape>/  as 64x64 silhouettes - exactly what the CNN sees.
Tip: collect 150+ per shape, and move/rotate/tilt the shape between captures.
"""
import argparse
import os
import time

import cv2
import numpy as np

from shape_common import LABELS, analyse

KEYS = {ord(str(i + 1)): LABELS[i] for i in range(len(LABELS))}


def open_source(src):
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def rotate(frame, deg):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def next_index(folder):
    nums = [int(f.split(".")[0]) for f in os.listdir(folder)
            if f.split(".")[0].isdigit()]
    return (max(nums) + 1) if nums else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="DroidCam URL (http://IP:4747/video) or camera index")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate the incoming video (phone held sideways?)")
    ap.add_argument("--flip", action="store_true", help="mirror the video")
    args = ap.parse_args()

    for label in LABELS:
        os.makedirs(os.path.join(args.out, label), exist_ok=True)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera source: %s" % args.source)

    current = LABELS[0]
    continuous = False
    paused = False
    last_save = 0.0
    counts = {l: len(os.listdir(os.path.join(args.out, l))) for l in LABELS}
    frame = None
    toast, toast_until = "", 0.0

    def save(label, sil):
        folder = os.path.join(args.out, label)
        cv2.imwrite(os.path.join(folder, "%05d.png" % next_index(folder)), sil)
        counts[label] += 1

    print(__doc__)
    while True:
        if not paused:
            ok, grabbed = cap.read()
            if not ok:
                print("No frame received - check the connection.")
                time.sleep(0.2)
                continue
            frame = rotate(grabbed, args.rotate)
            if args.flip:
                frame = cv2.flip(frame, 1)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sil, contour, (sw, sh) = analyse(gray)

        view = frame.copy()
        if contour is not None:
            fx, fy = frame.shape[1] / sw, frame.shape[0] / sh
            poly = np.round(contour * [fx, fy]).astype(np.int32)
            cv2.polylines(view, [poly], True, (0, 200, 0), 3)

        now = time.time()
        if contour is not None and continuous and not paused and now - last_save > 0.33:
            save(current, sil)
            last_save = now

        status = "[%s]  %s%s  saved: %s" % (
            current, "CONTINUOUS " if continuous else "",
            "PAUSED" if paused else "",
            " ".join("%s=%d" % (l[:3], counts[l]) for l in LABELS))
        cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4)
        cv2.putText(view, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        if contour is None:
            cv2.putText(view, "no shape found", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if now < toast_until:
            cv2.putText(view, toast, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 160, 255), 2)
        if sil is not None:
            thumb = cv2.cvtColor(cv2.resize(sil, (96, 96)), cv2.COLOR_GRAY2BGR)
            view[view.shape[0] - 100:view.shape[0] - 4, 4:100] = thumb

        cv2.imshow("Shape collector (1-5 pick, SPACE, C, P, Q)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in KEYS:
            current = KEYS[key]
        elif key == ord(" "):
            if sil is not None:
                save(current, sil)
                toast, toast_until = "saved %s" % current, now + 0.8
            else:
                toast, toast_until = "no shape to save", now + 0.8
        elif key == ord("c"):
            continuous = not continuous
        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("Totals:", counts)


if __name__ == "__main__":
    main()
