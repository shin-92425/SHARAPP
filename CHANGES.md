# SHARAPP v2: what changed and how to apply it

## Where the files go
| File in this zip | Copy to (your project) |
|---|---|
| `python/shape_common.py` | `python/shape_common.py` |
| `python/synth.py` | `python/synth.py` |
| `python/train.py` | `python/train.py` |
| `python/bench.py` | `python/bench.py` (new, optional benchmark tool) |
| `lib/services/shape_detector.dart` | `lib/services/shape_detector.dart` |
| `lib/screens/home_screen.dart` | `lib/screens/home_screen.dart` |

`collect_data.py` and your `dataset/` folder are untouched and keep working: `shape_common.py` still exports every name the old one did.

## Steps
1. Copy the files above.
2. `cd python` then `python train.py --real dataset`. Rendering the synthetic shapes takes several minutes. Read the block called **REAL held-out photos**: that is the first honest accuracy number you have had.
3. The new `shape_model.tflite` and a 6-line `labels.txt` (five shapes + `other`) are written to `assets/models/` automatically.
4. `flutter pub get` (no new packages), `flutter analyze`, then run on the phone.

Update the app and the model together. The pipeline changed, so an old model with the new app (or the reverse) will be mismatched.

## What changed

**Detection (`shape_common.py` + `shape_detector.dart`, kept identical)**
- Frames are shrunk by area averaging (`INTER_AREA`) instead of picking single pixels.
- The lighting is flattened (divide by a heavily blurred copy) before the Otsu threshold, so shadows and gradients stop breaking it.
- Contours that are slivers (paper edges) or wispy/hollow are rejected (solidity and elongation checks).
- The best **3** contours are kept. The CNN looks at each and the most convincing real shape wins, so a big object near the shape no longer steals the detection.

**Training (`train.py`, `synth.py`)**
- Honest validation: real photos are split into train / held-out *before* augmenting, in contiguous blocks (consecutive frames are near-duplicates). The old script validated on images it had trained on.
- Real silhouettes are augmented by rotating on a roomy canvas, then re-fitting to the bounding box like the app does (the old jitter clipped corners and kept the wrong size). Perspective and mirror flips added.
- New class `other` (pentagon, hexagon, cross, arrow, blobs ... plus distractor contours), so unknown shapes are rejected instead of forced into one of the five.
- Synthetic frames are now phone-shaped (480x720 portrait) with shadows, low contrast, outline-only and rounded shapes, motion blur, paper texture and distractors.
- Reports: synthetic vs real accuracy, confusion matrix, accuracy at the app's 0.80 confidence threshold. Early stopping keeps the best epoch.

**App (`home_screen.dart`)**
- The label on screen is a vote over the last 5 frames (a new label needs 3 votes), so it no longer flickers. Auto-save uses the voted label.

## Measured here (synthetic frames; `python bench.py` reproduces it)
Share of frames where the pipeline found the right contour (IoU of at least 0.8), 300 frames per condition, same frames for both:

| Condition | Current pipeline | New pipeline |
|---|---|---|
| easy | 100% | 100% |
| gradient (strong lighting ramp) | 87% | 98% |
| shadow (hard-edged) | 65% | 93% |
| low contrast | 76% | 100% |
| clutter (distractors touching the shape) | 29% | 32% |
| small shapes | 100% | 100% |
| blur + noise | 100% | 100% |
| outline-only | 91% | 95% |
| **Average** | **81%** | **90%** |

- Big separate object outranking the shape: top-1 pick right 74% of the time; the right shape is among the top 3 in 100% of those frames. With a stand-in classifier the final answer went from 73% to 96%.
- Label voting, simulated: on a steady shape with 10% wrong frames, label changes went from 81 to 1 over 300 frames.

## Not verified, please read
- **The Dart code was not compiled** (no Flutter SDK here). I checked the opencv_dart calls against its published docs and ported the pure-Dart geometry and rotation to Python to compare with OpenCV and numpy, but run `flutter analyze` first. If it flags a call, tell me the exact message.
- **The CNN was not trained here** (no TensorFlow). Your real accuracy comes from `train.py`'s *REAL held-out photos* block.
- The benchmark is synthetic. Clutter that touches the shape in the same tone cannot be separated by any threshold method; a plain background is still the best advice for users.

## Knobs if you want to tune
- `FLATTEN_DIV` (8): smaller = flatter lighting, but outline-only shapes suffer.
- `MIN_SOLIDITY` (0.35) and `MAX_ELONGATION` (4): the sliver/wisp filters.
- `TOP_K` (3): how many contours the CNN checks.
- `minConfidence` (0.80, in `train.py` and `shape_detector.dart`): raise it to be stricter.
- Optional: drop real photos of non-shapes into `dataset/other/` (64x64 PNGs); `train.py` picks them up.
