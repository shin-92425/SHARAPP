import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../services/shape_detector.dart';
import '../theme.dart';
import '../widgets/shape_painters.dart';
import 'home_screen.dart';

/// Opening screen: three shapes bounce while the model loads.
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late final AnimationController _bounce =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))
        ..repeat();

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    // load the model while the splash is showing, but never stay under 2.4 s
    final minimum = Future.delayed(const Duration(milliseconds: 2400));
    try {
      await ShapeDetector.instance.load();
    } catch (_) {
      // HomeScreen will report a friendly error if the model really is missing
    }
    await minimum;
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        transitionDuration: const Duration(milliseconds: 500),
        pageBuilder: (_, __, ___) => const HomeScreen(),
        transitionsBuilder: (_, anim, __, child) =>
            FadeTransition(opacity: anim, child: child),
      ),
    );
  }

  @override
  void dispose() {
    _bounce.dispose();
    super.dispose();
  }

  Widget _bouncer(ShapeKind kind, Color color, double phase) {
    return AnimatedBuilder(
      animation: _bounce,
      builder: (_, __) {
        final t = (_bounce.value + phase) % 1.0;
        final lift = math.sin(t * math.pi).abs() * 26;
        return Padding(
          padding: EdgeInsets.only(bottom: lift, top: 26 - lift),
          child: ShapeIcon(kind, size: 64, color: color),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _bouncer(ShapeKind.circle, AppColors.blue, 0.0),
                const SizedBox(width: 18),
                _bouncer(ShapeKind.triangle, AppColors.ink, 0.15),
                const SizedBox(width: 18),
                _bouncer(ShapeKind.square, AppColors.blue, 0.30),
              ],
            ),
            const SizedBox(height: 28),
            const Text(
              'Shape Buddy',
              style: TextStyle(fontSize: 40, fontWeight: FontWeight.w800, letterSpacing: 0.5),
            ),
            const SizedBox(height: 8),
            Text(
              "Let's find some shapes!",
              style: TextStyle(
                fontSize: 18,
                color: AppColors.ink.withValues(alpha: 0.7),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
