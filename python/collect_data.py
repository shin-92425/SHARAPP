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

Controls - click the buttons under the video, or press the keys
  Shape buttons   1-5     choose which shape you are showing
                          (1 circle, 2 heart, 3 square, 4 star, 5 triangle)
  Single capture  SPACE   save one picture now
  Continuous      C       on/off. While ON, a picture is saved about 3 times a
                          second for as long as a shape is seen (--interval)
  Pause / Resume  P       stop ALL saving, including continuous mode. The live
                          video keeps running so you can change the shape;
                          press again and continuous capture carries on
  Exit            Q/ESC   close the collector

The green/blue outline is what the app would draw. Pictures are saved to
dataset/<shape>/ as 64x64 silhouettes - exactly what the CNN sees.
Tip: collect 150+ per shape, and move/rotate/tilt the shape between captures.
"""
import argparse
import os
import time

import cv2
import numpy as np

from shape_common import LABELS, analyse

WIN = "Shape collector"
VIEW_W = 960                      # the video is shown this wide
BTN_H = 46
PAD = 8
PANEL_H = BTN_H * 2 + PAD * 3     # two rows of buttons under the video
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Same three colours as the app (BGR order): cream, ink, one blue.
CREAM = (238, 248, 255)
INK = (81, 63, 59)
BLUE = (166, 157, 79)
BLUE_SOFT = (240, 238, 220)
WHITE = (255, 255, 255)
RED = (60, 60, 210)


def open_source(src):
    return cv2.VideoCapture(int(src) if src.isdigit() else src)


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
            if f.endswith(".png") and f.split(".")[0].isdigit()]
    return (max(nums) + 1) if nums else 0


def key_action(key):
    """Keyboard key code -> the same action names the buttons use."""
    if key == 27:
        return "exit"
    ch = chr(key).lower() if 0 < key < 128 else ""
    if ch == "q":
        return "exit"
    if ch == " ":
        return "single"
    if ch == "c":
        return "continuous"
    if ch == "p":
        return "pause"
    for i, label in enumerate(LABELS):
        if ch == str(i + 1):
            return "shape:" + label
    return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def put_centered(img, text, rect, color, max_scale=0.62):
    """Centre text in a rectangle, shrinking it until it fits."""
    x0, y0, x1, y1 = rect
    avail = (x1 - x0) - 16
    scale = max_scale
    while scale > 0.35 and cv2.getTextSize(text, FONT, scale, 1)[0][0] > avail:
        scale -= 0.04
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    org = (x0 + ((x1 - x0) - tw) // 2, y0 + ((y1 - y0) + th) // 2)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw_button(img, rect, text, active):
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), BLUE if active else BLUE_SOFT, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 2)
    put_centered(img, text, rect, WHITE if active else INK)


def tag(img, text, x, y, fg, bg, right=False, scale=0.7):
    """A small filled label. With right=True, x is the right-hand edge."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    w = tw + 20
    x0 = x - w if right else x
    cv2.rectangle(img, (x0, y), (x0 + w, y + th + 16), bg, -1)
    cv2.putText(img, text, (x0 + 10, y + th + 8), FONT, scale, fg, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# The collector: state + actions (kept apart from the window so it can be tested)
# ---------------------------------------------------------------------------

class Collector:
    def __init__(self, out, interval=0.33):
        self.out = out
        self.interval = interval
        self.current = LABELS[0]
        self.continuous = False
        self.paused = False
        self.want_exit = False
        self.last_save = -1e9
        self.toast = ""
        self.toast_until = 0.0
        self.counts = {}
        for label in LABELS:
            folder = os.path.join(out, label)
            os.makedirs(folder, exist_ok=True)
            self.counts[label] = len([f for f in os.listdir(folder) if f.endswith(".png")])

    # -- saving -----------------------------------------------------------

    def _save(self, sil):
        folder = os.path.join(self.out, self.current)
        cv2.imwrite(os.path.join(folder, "%05d.png" % next_index(folder)), sil)
        self.counts[self.current] += 1

    def say(self, text, now):
        self.toast, self.toast_until = text, now + 1.2

    # -- actions (from buttons or keys) ---------------------------------------

    def act(self, action, sil, now):
        if action is None:
            return
        if action.startswith("shape:"):
            self.current = action[len("shape:"):]
            self.say("Collecting: %s" % self.current.capitalize(), now)
        elif action == "single":
            if self.paused:
                self.say("Paused - press Resume first", now)
            elif sil is None:
                self.say("No shape to save", now)
            else:
                self._save(sil)
                self.say("Saved %s  (%d)" % (self.current, self.counts[self.current]), now)
        elif action == "continuous":
            self.continuous = not self.continuous
            self.say("Continuous capture ON" if self.continuous
                     else "Continuous capture OFF", now)
        elif action == "pause":
            self.paused = not self.paused
            self.say("Paused - nothing is saved" if self.paused else "Resumed", now)
        elif action == "exit":
            self.want_exit = True

    def tick(self, sil, now):
        """Continuous capture. Returns True when it saved a picture."""
        if (self.continuous and not self.paused and sil is not None
                and now - self.last_save >= self.interval):
            self._save(sil)
            self.last_save = now
            return True
        return False

    # -- buttons -----------------------------------------------------------

    def layout(self, width, y_top):
        """Button rectangles (x0, y0, x1, y1) in window coordinates."""
        rows = [["shape:" + l for l in LABELS],
                ["single", "continuous", "pause", "exit"]]
        rects = {}
        for r, ids in enumerate(rows):
            n = len(ids)
            bw = (width - PAD * (n + 1)) // n
            top = y_top + PAD + r * (BTN_H + PAD)
            for i, bid in enumerate(ids):
                x0 = PAD + i * (bw + PAD)
                rects[bid] = (x0, top, x0 + bw, top + BTN_H)
        return rects

    @staticmethod
    def hit(x, y, rects):
        for bid, (x0, y0, x1, y1) in rects.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return bid
        return None

    def label(self, bid):
        if bid.startswith("shape:"):
            name = bid[len("shape:"):]
            return "%d  %s  %d" % (LABELS.index(name) + 1, name.capitalize(),
                                   self.counts[name])
        if bid == "single":
            return "Single capture  [Space]"
        if bid == "continuous":
            return "Continuous: %s  [C]" % ("ON" if self.continuous else "OFF")
        if bid == "pause":
            return ("Resume" if self.paused else "Pause") + "  [P]"
        return "Exit  [Q]"

    def active(self, bid):
        if bid.startswith("shape:"):
            return bid[len("shape:"):] == self.current
        if bid == "continuous":
            return self.continuous
        if bid == "pause":
            return self.paused
        return False

    # -- picture -----------------------------------------------------------

    def compose(self, frame, contour, size, sil, now):
        """Video + overlays + button panel. Returns (picture, button rectangles)."""
        h, w = frame.shape[:2]
        vh = int(round(h * VIEW_W / w))
        view = cv2.resize(frame, (VIEW_W, vh))

        if contour is not None:
            sw, sh = size
            poly = np.round(contour * [VIEW_W / sw, vh / sh]).astype(np.int32)
            cv2.polylines(view, [poly], True, WHITE, 9, cv2.LINE_AA)
            cv2.polylines(view, [poly], True, BLUE, 5, cv2.LINE_AA)

        tag(view, "Collecting: %s" % self.current.capitalize(), 12, 12, WHITE, INK)
        if self.paused:
            tag(view, "PAUSED - not saving", VIEW_W - 12, 12, WHITE, INK, right=True)
        elif self.continuous:
            if sil is not None:
                tag(view, "REC - saving", VIEW_W - 12, 12, WHITE, RED, right=True)
            else:
                tag(view, "Continuous: waiting for a shape", VIEW_W - 12, 12,
                    WHITE, INK, right=True)
        if sil is None and not self.paused:
            tag(view, "no shape found", 12, 56, WHITE, RED, scale=0.6)
        if now < self.toast_until:
            tag(view, self.toast, 12, vh - 56, WHITE, INK)
        if sil is not None:
            thumb = cv2.cvtColor(cv2.resize(sil, (96, 96)), cv2.COLOR_GRAY2BGR)
            view[vh - 108:vh - 12, VIEW_W - 108:VIEW_W - 12] = thumb
            cv2.rectangle(view, (VIEW_W - 108, vh - 108), (VIEW_W - 12, vh - 12), BLUE, 2)

        canvas = np.full((vh + PANEL_H, VIEW_W, 3), CREAM, np.uint8)
        canvas[:vh] = view
        rects = self.layout(VIEW_W, vh)
        for bid, rect in rects.items():
            draw_button(canvas, rect, self.label(bid), self.active(bid))
        return canvas, rects


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="DroidCam URL (http://IP:4747/video) or camera index")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--interval", type=float, default=0.33,
                    help="seconds between pictures in continuous mode (default 0.33)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate the incoming video (phone held sideways?)")
    ap.add_argument("--flip", action="store_true", help="mirror the video")
    args = ap.parse_args()

    col = Collector(args.out, args.interval)
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera source: %s" % args.source)

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    print(__doc__)
    rects = {}
    fails = 0
    while not col.want_exit:
        ok, grabbed = cap.read()
        if not ok:
            fails += 1
            if fails % 20 == 1:
                print("No frame received - check the connection.")
            if key_action(cv2.waitKey(100) & 0xFF) == "exit":
                break
            continue
        fails = 0

        frame = rotate(grabbed, args.rotate)
        if args.flip:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sil, contour, size = analyse(gray)
        now = time.time()

        while clicks:                                   # button clicks
            x, y = clicks.pop(0)
            col.act(col.hit(x, y, rects), sil, now)
        col.tick(sil, now)                              # continuous capture

        canvas, rects = col.compose(frame, contour, size, sil, now)
        cv2.imshow(WIN, canvas)

        col.act(key_action(cv2.waitKey(1) & 0xFF), sil, now)   # keyboard
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break                                       # window closed with the X

    cap.release()
    cv2.destroyAllWindows()
    print("Totals:", col.counts)


if __name__ == "__main__":
    main()
