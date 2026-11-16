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

/// Mirror of python/shape_common.py. Keep the constants and steps identical:
///   shrink -> 3x3 blur -> Otsu (both ways) -> contours -> pick best ->
///   64x64 silhouette -> CNN.
class ShapeDetector {
  ShapeDetector._();
  static final ShapeDetector instance = ShapeDetector._();

  static const int maxSide = 200;
  static const int borderMargin = 2;
  static const double minAreaFrac = 0.02;
  static const double maxAreaFrac = 0.60;
  static const int sil = 64;
  static const int silPad = 6;

  /// Below this the app says "I'm not sure" instead of guessing.
  static const double minConfidence = 0.80;

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
    final blurred = _blur(g.pixels, g.w, g.h);
    final contour = _findShape(blurred, g.w, g.h);
    if (contour == null) return null;

    final silPts = _silhouettePoints(contour);
    final silImg = _fill(silPts);

    // CNN input: [1, 64, 64, 1] floats in 0..1
    final input = List.generate(
      1,
      (_) => List.generate(
        sil,
        (y) => List.generate(sil, (x) => [silImg[y * sil + x] / 255.0]),
      ),
    );
    final output = List.generate(1, (_) => List<double>.filled(_labels.length, 0.0));
    interpreter.run(input, output);

    final probs = output[0];
    var best = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) best = i;
    }
    if (probs[best] < minConfidence) return null;

    // thin the outline out so drawing stays cheap
    final step = math.max(1, (contour.length / 160).ceil());
    final outline = <Offset>[];
    for (var i = 0; i < contour.length; i += step) {
      outline.add(Offset((contour[i].dx + 0.5) / g.w, (contour[i].dy + 0.5) / g.h));
    }

    return DetectionResult(
      label: _labels[best],
      confidence: probs[best],
      outline: outline,
      silhouette: silPts,
    );
  }

  // ---------------------------------------------------------------------
  // Step 1: Y (brightness) plane -> small upright grayscale picture
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
    if (scale < 1.0) {
      ow = math.max(1, (rw * scale).floor());
      oh = math.max(1, (rh * scale).floor());
    }

    final out = Uint8List(ow * oh);
    for (var oy = 0; oy < oh; oy++) {
      final yd = (oy * rh) ~/ oh;
      for (var ox = 0; ox < ow; ox++) {
        final xd = (ox * rw) ~/ ow;
        int xs, ys;
        switch (quarter) {
          case 1:
            xs = yd;
            ys = h - 1 - xd;
            break;
          case 2:
            xs = w - 1 - xd;
            ys = h - 1 - yd;
            break;
          case 3:
            xs = w - 1 - yd;
            ys = xd;
            break;
          default:
            xs = xd;
            ys = yd;
        }
        out[oy * ow + ox] = bytes[ys * stride + xs];
      }
    }
    return _Gray(out, ow, oh);
  }

  // ---------------------------------------------------------------------
  // Step 2: 3x3 box blur (reflect-101 borders, like cv2.blur)
  // ---------------------------------------------------------------------

  static int _reflect(int i, int n) {
    if (n < 2) return 0;
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
  }

  Uint8List _blur(Uint8List src, int w, int h) {
    final dst = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var sum = 0;
        for (var dy = -1; dy <= 1; dy++) {
          final yy = _reflect(y + dy, h);
          for (var dx = -1; dx <= 1; dx++) {
            sum += src[yy * w + _reflect(x + dx, w)];
          }
        }
        dst[y * w + x] = (sum + 4) ~/ 9;
      }
    }
    return dst;
  }

  // ---------------------------------------------------------------------
  // Steps 3-5: OpenCV threshold + contours, then choose the best contour
  // ---------------------------------------------------------------------

  List<Offset>? _findShape(Uint8List gray, int w, int h) {
    final frameArea = (w * h).toDouble();
    final cx = w / 2.0, cy = h / 2.0;
    final halfDiag = math.sqrt(cx * cx + cy * cy);

    List<Offset>? best;
    var bestScore = 0.0;

    final mat = cv.Mat.fromList(h, w, cv.MatType.CV_8UC1, gray);
    try {
      for (final inverted in const [false, true]) {
        final mode = inverted ? cv.THRESH_BINARY_INV : cv.THRESH_BINARY;
        final (_, mask) = cv.threshold(mat, 0, 255, mode | cv.THRESH_OTSU);
        final (contours, hierarchy) =
            cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        for (final c in contours) {
          final pts = c.toList();
          if (pts.length < 5) continue;

          var minX = 1 << 30, minY = 1 << 30, maxX = -1, maxY = -1;
          for (final p in pts) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
          }

          final area = _area(pts);
          if (area < minAreaFrac * frameArea || area > maxAreaFrac * frameArea) {
            continue;
          }
          if (minX <= borderMargin ||
              minY <= borderMargin ||
              maxX >= w - 1 - borderMargin ||
              maxY >= h - 1 - borderMargin) {
            continue; // touches the edge: it's the background, not a shape
          }

          final mx = (minX + maxX) / 2.0, my = (minY + maxY) / 2.0;
          final dist = math.sqrt((mx - cx) * (mx - cx) + (my - cy) * (my - cy)) / halfDiag;
          final score = area * (1.0 - 0.5 * dist);
          if (score > bestScore) {
            bestScore = score;
            best = [for (final p in pts) Offset(p.x.toDouble(), p.y.toDouble())];
          }
        }
        mask.dispose();
        hierarchy.dispose();
      }
    } finally {
      mat.dispose();
    }
    return best;
  }

  /// Shoelace area of a contour.
  double _area(List<cv.Point> pts) {
    var s = 0.0;
    for (var i = 0; i < pts.length; i++) {
      final a = pts[i], b = pts[(i + 1) % pts.length];
      s += a.x * b.y - b.x * a.y;
    }
    return s.abs() / 2.0;
  }

  // ---------------------------------------------------------------------
  // Step 6: contour -> 64x64 filled silhouette
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
