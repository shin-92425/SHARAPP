import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../theme.dart';

enum ShapeKind { circle, heart, square, star, triangle }

ShapeKind? shapeKindFromLabel(String label) {
  for (final k in ShapeKind.values) {
    if (k.name == label) return k;
  }
  return null;
}

String prettyLabel(String label) =>
    label.isEmpty ? label : label[0].toUpperCase() + label.substring(1);

/// A tidy path for each shape, drawn inside [r].
Path shapePath(ShapeKind kind, Rect r) {
  final w = r.width, h = r.height;
  Offset pt(double x, double y) => Offset(r.left + x * w, r.top + y * h);
  final path = Path();
  switch (kind) {
    case ShapeKind.circle:
      path.addOval(Rect.fromLTWH(r.left + w * 0.06, r.top + h * 0.06, w * 0.88, h * 0.88));
      break;
    case ShapeKind.square:
      path.addRRect(RRect.fromRectAndRadius(
        Rect.fromLTWH(r.left + w * 0.1, r.top + h * 0.1, w * 0.8, h * 0.8),
        Radius.circular(w * 0.1),
      ));
      break;
    case ShapeKind.triangle:
      final a = pt(0.5, 0.1), b = pt(0.95, 0.88), c = pt(0.05, 0.88);
      path
        ..moveTo(a.dx, a.dy)
        ..lineTo(b.dx, b.dy)
        ..lineTo(c.dx, c.dy)
        ..close();
      break;
    case ShapeKind.star:
      final cx = r.center.dx, cy = r.center.dy + h * 0.03;
      final outer = math.min(w, h) * 0.5;
      final inner = outer * 0.46;
      for (var i = 0; i < 10; i++) {
        final rad = i.isEven ? outer : inner;
        final ang = -math.pi / 2 + i * math.pi / 5;
        final x = cx + rad * math.cos(ang), y = cy + rad * math.sin(ang);
        i == 0 ? path.moveTo(x, y) : path.lineTo(x, y);
      }
      path.close();
      break;
    case ShapeKind.heart:
      final s = pt(0.5, 0.92);
      path.moveTo(s.dx, s.dy);
      path.cubicTo(pt(0.0, 0.60).dx, pt(0.0, 0.60).dy, pt(0.0, 0.18).dx,
          pt(0.0, 0.18).dy, pt(0.26, 0.14).dx, pt(0.26, 0.14).dy);
      path.cubicTo(pt(0.40, 0.12).dx, pt(0.40, 0.12).dy, pt(0.48, 0.22).dx,
          pt(0.48, 0.22).dy, pt(0.5, 0.30).dx, pt(0.5, 0.30).dy);
      path.cubicTo(pt(0.52, 0.22).dx, pt(0.52, 0.22).dy, pt(0.60, 0.12).dx,
          pt(0.60, 0.12).dy, pt(0.74, 0.14).dx, pt(0.74, 0.14).dy);
      path.cubicTo(pt(1.0, 0.18).dx, pt(1.0, 0.18).dy, pt(1.0, 0.60).dx,
          pt(1.0, 0.60).dy, s.dx, s.dy);
      path.close();
      break;
  }
  return path;
}

/// Paints one of the five shapes (used on the splash, labels and empty states).
class ShapeIconPainter extends CustomPainter {
  final ShapeKind kind;
  final Color color;
  final bool filled;
  final double strokeWidth;

  const ShapeIconPainter(
    this.kind,
    this.color, {
    this.filled = true,
    this.strokeWidth = 4,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final path = shapePath(kind, Offset.zero & size);
    final paint = Paint()
      ..color = color
      ..style = filled ? PaintingStyle.fill : PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;
    canvas.drawPath(path, paint);
    if (filled) {
      canvas.drawPath(
        path,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = size.width * 0.06
          ..strokeJoin = StrokeJoin.round,
      );
    }
  }

  @override
  bool shouldRepaint(ShapeIconPainter old) =>
      old.kind != kind || old.color != color || old.filled != filled;
}

class ShapeIcon extends StatelessWidget {
  final ShapeKind kind;
  final double size;
  final Color color;
  final bool filled;

  const ShapeIcon(
    this.kind, {
    super.key,
    this.size = 40,
    this.color = AppColors.blue,
    this.filled = true,
  });

  @override
  Widget build(BuildContext context) => SizedBox(
        width: size,
        height: size,
        child: CustomPaint(painter: ShapeIconPainter(kind, color, filled: filled)),
      );
}

/// Draws the recognised shape's outline on top of the camera preview.
/// Points are 0..1 across the preview.
class OutlinePainter extends CustomPainter {
  final List<Offset> points;
  const OutlinePainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 3) return;
    final path = Path()
      ..moveTo(points.first.dx * size.width, points.first.dy * size.height);
    for (var i = 1; i < points.length; i++) {
      path.lineTo(points[i].dx * size.width, points[i].dy * size.height);
    }
    path.close();

    canvas.drawPath(
      path,
      Paint()
        ..color = AppColors.blue.withValues(alpha: 0.14)
        ..style = PaintingStyle.fill,
    );
    // white edge underneath so the line shows on dark and light backgrounds
    canvas.drawPath(
      path,
      Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 10
        ..strokeJoin = StrokeJoin.round,
    );
    canvas.drawPath(
      path,
      Paint()
        ..color = AppColors.blue
        ..style = PaintingStyle.stroke
        ..strokeWidth = 5.5
        ..strokeJoin = StrokeJoin.round,
    );
  }

  @override
  bool shouldRepaint(OutlinePainter old) => old.points != points;
}

/// Picture saved to history: the shape's outline on a cream card.
/// [silhouette] points are in 0..[space] coordinates.
Future<Uint8List> renderCapturePng(List<Offset> silhouette, {double space = 64}) async {
  const size = 256.0;
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder, const Rect.fromLTWH(0, 0, size, size));
  canvas.drawRect(
    const Rect.fromLTWH(0, 0, size, size),
    Paint()..color = AppColors.cream,
  );

  final k = size / space;
  final path = Path();
  for (var i = 0; i < silhouette.length; i++) {
    final p = silhouette[i] * k;
    i == 0 ? path.moveTo(p.dx, p.dy) : path.lineTo(p.dx, p.dy);
  }
  path.close();

  canvas.drawPath(path, Paint()..color = AppColors.blueSoft);
  canvas.drawPath(
    path,
    Paint()
      ..color = AppColors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 8
      ..strokeJoin = StrokeJoin.round,
  );

  final image = await recorder.endRecording().toImage(size.toInt(), size.toInt());
  final data = await image.toByteData(format: ui.ImageByteFormat.png);
  return data!.buffer.asUint8List();
}
