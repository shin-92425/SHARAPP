import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';

/// What the detector found in one camera frame.
class DetectionResult {
  final String label;
  final double confidence;

  /// Outline of the shape, each point 0..1 across the upright preview.
  final List<Offset> outline;

  /// Same outline mapped into the 64x64 silhouette square (for thumbnails).
  final List<Offset> silhouette;

  const DetectionResult({
    required this.label,
    required this.confidence,
    required this.outline,
    required this.silhouette,
  });
}

/// Mirror of python/shape_common.py (version 2). Keep the constants and steps
/// identical, and retrain the model after changing either side:
///
///   shrink (area average) -> 3x3 Gaussian blur -> flatten the lighting ->
///   Otsu (both ways) -> contours -> keep sensible ones -> top 3 ->
///   64x64 silhouette each -> CNN on each -> keep the most convincing shape.
class ShapeDetector {
  ShapeDetector._();
  static final ShapeDetector instance = ShapeDetector._();

  static const int maxSide = 200;
  static const int borderMargin = 2;
  static const double minAreaFrac = 0.02;
  static const double maxAreaFrac = 0.60;
  static const int sil = 64;
  static const int silPad = 6;

  static const double flattenDiv = 8.0; // flatten blur sigma = long side / this
  static const double minSolidity = 0.35; // contour area / convex hull area
  static const double maxElongation = 4.0; // long side / short side of tightest box
  static const int topK = 3; // candidates the CNN gets to look at

  /// Below this the app says "I'm not sure" instead of guessing.
  static const double minConfidence = 0.80;

  /// The CNN's "this is not one of my five shapes" class.
  static const String otherLabel = 'other';

  Interpreter? _interpreter;
  List<String> _labels = const [];
  Future<void>? _loading;

  Future<void> load() => _loading ??= _doLoad();

  Future<void> _doLoad() async {
    _interpreter = await Interpreter.fromAsset('assets/models/shape_model.tflite');
    final raw = await rootBundle.loadString('assets/models/labels.txt');
    _labels = raw.split('\n').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
  }

  bool get isReady => _interpreter != null;

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loading = null;
  }

  // ---------------------------------------------------------------------
  // Public entry point
  // ---------------------------------------------------------------------

  /// [sensorOrientation] is CameraDescription.sensorOrientation (0/90/180/270).
  DetectionResult? detect(CameraImage image, int sensorOrientation) {
    final interpreter = _interpreter;
    if (interpreter == null) return null;

    final g = _prepare(image, sensorOrientation);
    final candidates = _findCandidates(g.pixels, g.w, g.h);
    if (candidates.isEmpty) return null;

    // Ask the CNN about every candidate. A candidate only counts if the CNN's
    // top answer is one of the real shapes (not "other") and it is confident;
    // among those, the most confident wins. This lets a real shape beat a
    // bigger distractor that the size ranking put first.
    List<Offset>? bestContour;
    List<Offset>? bestSilPts;
    var bestLabel = '';
    var bestProb = 0.0;
    for (final contour in candidates) {
      final silPts = _silhouettePoints(contour);
      final probs = _classify(interpreter, _fill(silPts));

      var top = 0;
      for (var i = 1; i < probs.length; i++) {
        if (probs[i] > probs[top]) top = i;
      }
      final label = _labels[top];
      if (label == otherLabel || probs[top] < minConfidence) continue;
      if (probs[top] > bestProb) {
        bestProb = probs[top];
        bestLabel = label;
        bestContour = contour;
        bestSilPts = silPts;
      }
    }
    if (bestContour == null || bestSilPts == null) return null;

    // thin the outline out so drawing stays cheap
    final step = math.max(1, (bestContour.length / 160).ceil());
    final outline = <Offset>[];
    for (var i = 0; i < bestContour.length; i += step) {
      outline.add(Offset((bestContour[i].dx + 0.5) / g.w, (bestContour[i].dy + 0.5) / g.h));
    }

    return DetectionResult(
      label: bestLabel,
      confidence: bestProb,
      outline: outline,
      silhouette: bestSilPts,
    );
  }

  /// CNN input: [1, 64, 64, 1] floats in 0..1  ->  one probability per label.
  List<double> _classify(Interpreter interpreter, Uint8List silImg) {
    final input = List.generate(
      1,
      (_) => List.generate(
        sil,
        (y) => List.generate(sil, (x) => [silImg[y * sil + x] / 255.0]),
      ),
    );
    final output = List.generate(1, (_) => List<double>.filled(_labels.length, 0.0));
    interpreter.run(input, output);
    return output[0];
  }

  // ---------------------------------------------------------------------
  // Step 1: Y (brightness) plane -> small upright grayscale picture
  //         (area-averaged shrink, same as cv2.resize INTER_AREA in Python)
  // ---------------------------------------------------------------------

  _Gray _prepare(CameraImage image, int sensorOrientation) {
    final plane = image.planes[0];
    final bytes = plane.bytes;
    final stride = plane.bytesPerRow;
    final w = image.width;
    final h = image.height;

    // Clockwise quarter-turns needed to make the picture upright. Frames that
    // already arrive portrait-shaped are treated as upright.
    final quarter = h > w ? 0 : (sensorOrientation ~/ 90) % 4;
    final rw = quarter.isOdd ? h : w; // size after turning upright
    final rh = quarter.isOdd ? w : h;

    var ow = rw, oh = rh;
    final scale = maxSide / math.max(rw, rh);
    final shrinking = scale < 1.0;
    if (shrinking) {
      ow = math.max(1, (rw * scale).floor());
      oh = math.max(1, (rh * scale).floor());
    }

    // The Y plane without its row padding.
    Uint8List tight;
    if (stride == w) {
      tight = Uint8List.sublistView(bytes, 0, w * h);
    } else {
      tight = Uint8List(w * h);
      for (var y = 0; y < h; y++) {
        tight.setRange(y * w, (y + 1) * w, bytes, y * stride);
      }
    }

    // Shrink first (while still sideways), turn upright afterwards: the same
    // result as turning first, for a fraction of the work.
    final dw = quarter.isOdd ? oh : ow; // size of the shrunken, sideways picture
    final dh = quarter.isOdd ? ow : oh;
    Uint8List side;
    if (shrinking) {
      final src = cv.Mat.fromList(h, w, cv.MatType.CV_8UC1, tight);
      final small = cv.resize(src, (dw, dh), interpolation: cv.INTER_AREA);
      side = Uint8List.fromList(small.data);
      small.dispose();
      src.dispose();
    } else {
      side = tight;
    }

    if (quarter == 0) return _Gray(side, ow, oh);

    final out = Uint8List(ow * oh);
    for (var oy = 0; oy < oh; oy++) {
      for (var ox = 0; ox < ow; ox++) {
        int xs, ys;
        switch (quarter) {
          case 1:
            xs = oy;
            ys = dh - 1 - ox;
            break;
          case 2:
            xs = dw - 1 - ox;
            ys = dh - 1 - oy;
            break;
          default: // 3
            xs = dw - 1 - oy;
            ys = ox;
        }
        out[oy * ow + ox] = side[ys * dw + xs];
      }
    }
    return _Gray(out, ow, oh);
  }

  // ---------------------------------------------------------------------
  // Steps 2-3: 3x3 Gaussian blur, then divide out the slowly-changing
  //            lighting so shadows and gradients stop fooling the threshold.
  // ---------------------------------------------------------------------

  Uint8List _flatten(Uint8List gray, int w, int h) {
    final src = cv.Mat.fromList(h, w, cv.MatType.CV_8UC1, gray);
    final blurMat = cv.gaussianBlur(src, (3, 3), 0);
    final bgMat = cv.gaussianBlur(blurMat, (0, 0), math.max(w, h) / flattenDiv);
    final b = Uint8List.fromList(blurMat.data);
    final bg = Uint8List.fromList(bgMat.data);
    bgMat.dispose();
    blurMat.dispose();
    src.dispose();

    final out = Uint8List(w * h);
    for (var i = 0; i < out.length; i++) {
      final d = bg[i] < 1 ? 1 : bg[i];
      final v = b[i] / d * 128.0;
      out[i] = v >= 255.0 ? 255 : v.toInt(); // truncates, like numpy astype
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Steps 4-6: Otsu threshold + contours, filter, rank, keep the best few
  // ---------------------------------------------------------------------

  List<List<Offset>> _findCandidates(Uint8List gray, int w, int h) {
    final frameArea = (w * h).toDouble();
    final cx = w / 2.0, cy = h / 2.0;
    final halfDiag = math.sqrt(cx * cx + cy * cy);

    final flat = _flatten(gray, w, h);
    final found = <_Cand>[];

    final mat = cv.Mat.fromList(h, w, cv.MatType.CV_8UC1, flat);
    try {
      for (final inverted in const [false, true]) {
        final mode = inverted ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY;
        final (_, mask) = cv.threshold(mat, 0, 255, mode | cv.THRESH_OTSU);
        final (contours, hierarchy) =
            cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        for (final c in contours) {
          final raw = c.toList();
          if (raw.length < 5) continue;
          final pts = <Offset>[
            for (final p in raw) Offset(p.x.toDouble(), p.y.toDouble()),
          ];

          var minX = double.infinity, minY = double.infinity;
          var maxX = -double.infinity, maxY = -double.infinity;
          for (final p in pts) {
            if (p.dx < minX) minX = p.dx;
            if (p.dy < minY) minY = p.dy;
            if (p.dx > maxX) maxX = p.dx;
            if (p.dy > maxY) maxY = p.dy;
          }

          final area = _polyArea(pts);
          if (area < minAreaFrac * frameArea || area > maxAreaFrac * frameArea) {
            continue;
          }
          if (minX <= borderMargin ||
              minY <= borderMargin ||
              maxX >= w - 1 - borderMargin ||
              maxY >= h - 1 - borderMargin) {
            continue; // touches the edge: it's the background, not a shape
          }
          if (!_qualityOk(pts, area)) continue;

          final mx = (minX + maxX) / 2.0, my = (minY + maxY) / 2.0;
          final dist = math.sqrt((mx - cx) * (mx - cx) + (my - cy) * (my - cy)) / halfDiag;
          found.add(_Cand(area * (1.0 - 0.5 * dist), minX, minY, maxX, maxY, pts));
        }
        mask.dispose();
        hierarchy.dispose();
      }
    } finally {
      mat.dispose();
    }

    found.sort((a, b) => b.score.compareTo(a.score));
    final kept = <_Cand>[];
    for (final c in found) {
      if (kept.any((k) => _bboxIou(c, k) > 0.85)) continue; // same blob, seen twice
      kept.add(c);
      if (kept.length >= topK) break;
    }
    return [for (final c in kept) c.pts];
  }

  /// Rejects slivers (paper edges, table seams) and wispy or hollow blobs.
  bool _qualityOk(List<Offset> pts, double area) {
    final hull = _convexHull(pts);
    if (hull.length < 3) return false;
    final hullArea = _polyArea(hull);
    if (hullArea <= 0 || area / hullArea < minSolidity) return false;
    final (rw, rh) = _minAreaRectSides(hull);
    final shortSide = math.min(rw, rh), longSide = math.max(rw, rh);
    if (shortSide < 1.0 || longSide / shortSide > maxElongation) return false;
    return true;
  }

  double _bboxIou(_Cand a, _Cand b) {
    final ix0 = math.max(a.x0, b.x0), iy0 = math.max(a.y0, b.y0);
    final ix1 = math.min(a.x1, b.x1), iy1 = math.min(a.y1, b.y1);
    final inter = math.max(0.0, ix1 - ix0) * math.max(0.0, iy1 - iy0);
    final ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return ua > 0 ? inter / ua : 0.0;
  }

  /// Shoelace area of a polygon.
  double _polyArea(List<Offset> pts) {
    var s = 0.0;
    for (var i = 0; i < pts.length; i++) {
      final a = pts[i], b = pts[(i + 1) % pts.length];
      s += a.dx * b.dy - b.dx * a.dy;
    }
    return s.abs() / 2.0;
  }

  /// Convex hull (Andrew's monotone chain).
  List<Offset> _convexHull(List<Offset> pts) {
    final p = List<Offset>.of(pts)
      ..sort((a, b) => a.dx != b.dx ? a.dx.compareTo(b.dx) : a.dy.compareTo(b.dy));
    if (p.length < 3) return p;
    double cross(Offset o, Offset a, Offset b) =>
        (a.dx - o.dx) * (b.dy - o.dy) - (a.dy - o.dy) * (b.dx - o.dx);

    final lower = <Offset>[];
    for (final q in p) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower.last, q) <= 0) {
        lower.removeLast();
      }
      lower.add(q);
    }
    final upper = <Offset>[];
    for (final q in p.reversed) {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper.last, q) <= 0) {
        upper.removeLast();
      }
      upper.add(q);
    }
    lower.removeLast();
    upper.removeLast();
    return [...lower, ...upper];
  }

  /// Side lengths of the smallest rectangle (any angle) around a convex hull.
  /// One side of that rectangle always lies along a hull edge, so try each.
  (double, double) _minAreaRectSides(List<Offset> hull) {
    var bestArea = double.infinity;
    var bestW = 0.0, bestH = 0.0;
    final n = hull.length;
    for (var i = 0; i < n; i++) {
      final a = hull[i], b = hull[(i + 1) % n];
      final dx = b.dx - a.dx, dy = b.dy - a.dy;
      final len = math.sqrt(dx * dx + dy * dy);
      if (len == 0) continue;
      final ux = dx / len, uy = dy / len; // along the edge
      final nx = -uy, ny = ux; // across it
      var minU = double.infinity, maxU = -double.infinity;
      var minN = double.infinity, maxN = -double.infinity;
      for (final p in hull) {
        final u = p.dx * ux + p.dy * uy;
        final v = p.dx * nx + p.dy * ny;
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minN) minN = v;
        if (v > maxN) maxN = v;
      }
      final rw = maxU - minU, rh = maxN - minN;
      if (rw * rh < bestArea) {
        bestArea = rw * rh;
        bestW = rw;
        bestH = rh;
      }
    }
    return (bestW, bestH);
  }

  // ---------------------------------------------------------------------
  // Step 7: contour -> 64x64 filled silhouette
  // ---------------------------------------------------------------------

  List<Offset> _silhouettePoints(List<Offset> c) {
    var minX = double.infinity, minY = double.infinity;
    var maxX = -double.infinity, maxY = -double.infinity;
    for (final p in c) {
      if (p.dx < minX) minX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy > maxY) maxY = p.dy;
    }
    final bw = math.max(maxX - minX, 1.0);
    final bh = math.max(maxY - minY, 1.0);
    final s = (sil - 2 * silPad) / math.max(bw, bh);
    final cx = (minX + maxX) / 2.0, cy = (minY + maxY) / 2.0;
    return [
      for (final p in c) Offset((p.dx - cx) * s + sil / 2.0, (p.dy - cy) * s + sil / 2.0)
    ];
  }

  /// Even-odd scanline fill, sampled at pixel centres (same as Python).
  Uint8List _fill(List<Offset> poly) {
    final img = Uint8List(sil * sil);
    final n = poly.length;
    final xs = <double>[];
    for (var y = 0; y < sil; y++) {
      final sy = y + 0.5;
      xs.clear();
      for (var i = 0; i < n; i++) {
        final a = poly[i], b = poly[(i + 1) % n];
        if ((a.dy <= sy) != (b.dy <= sy)) {
          xs.add(a.dx + (sy - a.dy) * (b.dx - a.dx) / (b.dy - a.dy));
        }
      }
      xs.sort();
      for (var k = 0; k + 1 < xs.length; k += 2) {
        final x0 = (xs[k] - 0.5).ceil().clamp(0, sil).toInt();
        final x1 = (xs[k + 1] - 0.5).ceil().clamp(0, sil).toInt();
        for (var x = x0; x < x1; x++) {
          img[y * sil + x] = 255;
        }
      }
    }
    return img;
  }
}

class _Gray {
  final Uint8List pixels;
  final int w;
  final int h;
  const _Gray(this.pixels, this.w, this.h);
}

/// One contour that passed the filters, with its ranking score and bounding box.
class _Cand {
  final double score;
  final double x0, y0, x1, y1;
  final List<Offset> pts;
  const _Cand(this.score, this.x0, this.y0, this.x1, this.y1, this.pts);
}
