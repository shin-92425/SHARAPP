"""
Run once, after `flutter create .`, from the project root:

    python tools/apply_android_config.py

It
  * adds the CAMERA permission and names the app "Shape Buddy"
  * sets minSdk to 26 (needed by TensorFlow Lite)
  * replaces the launcher icon (no more Flutter logo)
  * makes the Android launch screen a plain cream colour (no Flutter logo)
It prints what it did, and tells you if a file was not in the shape it expected.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_icon  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
APP = os.path.join(ROOT, "android", "app")
MAIN = os.path.join(APP, "src", "main")

LAUNCH_XML = """<?xml version="1.0" encoding="utf-8"?>
<layer-list xmlns:android="http://schemas.android.com/apk/res/android">
    <item>
        <shape android:shape="rectangle">
            <solid android:color="#FFFFF8EE" />
        </shape>
    </item>
</layer-list>
"""


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def patch_manifest():
    path = os.path.join(MAIN, "AndroidManifest.xml")
    if not os.path.exists(path):
        return "MISSING  " + path + "  (run `flutter create .` first)"
    text = read(path)
    notes = []
    if "android.permission.CAMERA" not in text:
        text = text.replace(
            "<application",
            '<uses-permission android:name="android.permission.CAMERA"/>\n    <application',
            1,
        )
        notes.append("camera permission added")
    new, n = re.subn(r'android:label="[^"]*"', 'android:label="Shape Buddy"', text, count=1)
    if n:
        text = new
        notes.append("app name set")
    write(path, text)
    return "OK       AndroidManifest.xml: " + ", ".join(notes or ["already patched"])


def patch_gradle():
    for name in ("build.gradle.kts", "build.gradle"):
        path = os.path.join(APP, name)
        if not os.path.exists(path):
            continue
        text = read(path)
        patterns = [
            (r"minSdk\s*=\s*flutter\.minSdkVersion", "minSdk = 26"),          # kts
            (r"minSdkVersion\s+flutter\.minSdkVersion", "minSdkVersion 26"),   # groovy
            (r"minSdk\s+flutter\.minSdkVersion", "minSdkVersion 26"),          # groovy
            (r"minSdk\s*=\s*\d+", "minSdk = 26"),
            (r"minSdkVersion\s+\d+", "minSdkVersion 26"),
        ]
        for pat, rep in patterns:
            new, n = re.subn(pat, rep, text, count=1)
            if n:
                write(path, new)
                return "OK       %s: minSdk set to 26" % name
        return "CHECK    %s: could not find minSdk - set it to 26 by hand" % name
    return "MISSING  android/app/build.gradle(.kts)  (run `flutter create .` first)"


def patch_launch():
    res = os.path.join(MAIN, "res")
    for d in ("drawable", "drawable-v21"):
        write(os.path.join(res, d, "launch_background.xml"), LAUNCH_XML)
    return "OK       launch screen is now plain cream"


def patch_icons():
    res = os.path.join(MAIN, "res")
    make_icon.write_icons(res)
    return "OK       launcher icons replaced"


if __name__ == "__main__":
    if not os.path.isdir(MAIN):
        sys.exit("No android/ folder here. Run `flutter create .` in the project root first.")
    for step in (patch_manifest, patch_gradle, patch_launch, patch_icons):
        print(step())
