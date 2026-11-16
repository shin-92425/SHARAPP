# Shape Buddy

A child-friendly Flutter app that looks through the **back camera**, recognises
**circle, heart, square, star and triangle**, and draws an **outline** around the shape it finds.

Everything runs on the phone: OpenCV (`opencv_dart`) finds the shape, a small
TensorFlow CNN (exported to TensorFlow Lite) names it, and SQLite (`sqflite`)
stores the history.

## What's inside

| Folder | What it is |
|---|---|
| `lib/` | The Flutter app (splash, camera screen, history screen, detector, database) |
| `assets/models/` | `shape_model.tflite` + `labels.txt` (already trained) |
| `python/` | Laptop tools: DroidCam data collector, synthetic data, CNN training |
| `tools/` | Android setup helper (icon, launch screen, camera permission) |

## 1. Run the app (Android)

Needs Flutter **3.27 or newer** and an Android phone (Android 8.0 / API 26+).

```bash
flutter create --platforms=android shape_buddy      # makes a fresh project
# copy lib/ assets/ python/ tools/ pubspec.yaml from this folder INTO it (overwrite)
cd shape_buddy
rm test/widget_test.dart                            # the default test doesn't match this app
python tools/apply_android_config.py                # camera permission, minSdk 26, custom icon, cream launch screen
flutter pub get
flutter run                                         # phone plugged in, USB debugging on
```

`apply_android_config.py` is what removes the Flutter logo from the app icon and the
Android launch screen. The in-app splash screen is drawn with Dart and has no Flutter logo either.

The first build downloads the OpenCV binaries, so it needs internet and takes a while.

## 2. How to use it

* **Point the back camera at a shape.** A blue outline appears around it and its name is shown below the picture.
* **Snap!** (big round button) saves one picture of the shape that is currently recognised.
* **Auto** (left) is *continuous* mode. While it is on (blue), every shape the app is sure about is saved automatically, at most once every ~2.5 seconds. Tap again to turn it off.
* **Pause / Resume** (right) freezes the camera. Snap! still saves the frozen shape.
* **Exit** (top left) closes the app after asking "Close Shape Buddy?".
* **History** (top right) opens *My Shapes*: every saved shape with its name, date and time. Tap the bin to delete one (it asks first); the sweep icon at the top deletes all.

Colours: cream, dark ink and one soft blue. Nothing else.

## 3. Make it better with your own photos (DroidCam)

The model that ships was trained on generated shapes. For best results with your real
paper shapes, teach it with photos from your phone camera:

```bash
cd python
pip install -r requirements.txt

# phone on the same Wi-Fi with DroidCam open (it shows an address like 192.168.1.5)
python collect_data.py --source http://192.168.1.5:4747/video
# or USB / DroidCam client:  python collect_data.py --source 1   (try 0, 1, 2 ...)
```

In the video window: press `1`-`5` to say which shape you are showing
(1 circle, 2 heart, 3 square, 4 star, 5 triangle), `SPACE` = single capture,
`C` = continuous capture, `P` = pause, `Q` = exit. The green line is the
outline the app would draw. If the picture is sideways add `--rotate 90`.

Collect **150+ per shape**, turning and tilting the shape between captures, and
in a few different lighting conditions. Then retrain:

```bash
python train.py --real dataset
```

This rewrites `assets/models/shape_model.tflite`. Run the app again.

## 4. How recognition works

1. The camera's brightness image is shrunk to 200 px on its long side, turned upright and blurred.
2. OpenCV applies an Otsu threshold (both dark-on-light and light-on-dark) and finds contours.
3. The best contour is chosen: big enough, not touching the picture's edge, near the centre.
4. That contour becomes a 64x64 black-and-white silhouette, and the CNN names it.
5. Under 80% sure, the app says "Show me a shape!" instead of guessing.

Because the CNN only sees a silhouette, colour and material don't matter. What matters is
that the shape **stands out from its background** (dark on white paper is ideal; cut-outs
or toys on a plain table also work). A busy background can break step 2.

`python/shape_common.py` and `lib/services/shape_detector.dart` are mirrors of each other.
If you change a constant in one, change it in the other and retrain.

## Known limits

* Android only (the camera frame handling is written for Android's YUV format).
* Pentagons, ovals and so on will be pushed into the nearest of the five shapes if the app is >80% sure.
* History pictures are a clean outline drawing of the shape, not a photo of the camera view.
