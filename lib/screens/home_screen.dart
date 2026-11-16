import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/db_helper.dart';
import '../services/shape_detector.dart';
import '../theme.dart';
import '../widgets/shape_painters.dart';
import 'history_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  String? _error;
  bool _ready = false;

  bool _continuous = false; // auto-save shapes as they are recognised
  bool _paused = false; // frozen view
  bool _busy = false; // a frame is being processed
  bool _flash = false;
  bool _initializing = false;
  bool _inHistory = false;

  DetectionResult? _current;
  int _misses = 0;
  String? _stableLabel;
  int _stableCount = 0;
  DateTime _lastRun = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastAutoSave = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _releaseCamera(notify: false);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _releaseCamera();
    } else if (state == AppLifecycleState.resumed && !_inHistory) {
      _initCamera();
    }
  }

  // ---------------------------------------------------------------------
  // Camera (back camera only)
  // ---------------------------------------------------------------------

  Future<void> _initCamera() async {
    if (_initializing || _controller != null) return;
    _initializing = true;
    try {
      await _doInitCamera();
    } finally {
      _initializing = false;
    }
  }

  Future<void> _doInitCamera() async {
    if (mounted) {
      setState(() {
        _error = null;
        _ready = false;
        _paused = false;
        _current = null;
      });
    }
    try {
      await ShapeDetector.instance.load();
      final cameras = await availableCameras();
      final back =
          cameras.where((c) => c.lensDirection == CameraLensDirection.back).toList();
      if (back.isEmpty) {
        if (mounted) setState(() => _error = "I can't find the back camera on this phone.");
        return;
      }
      final c = CameraController(
        back.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await c.initialize();
      await c.lockCaptureOrientation(DeviceOrientation.portraitUp);
      if (!mounted) {
        await c.dispose();
        return;
      }
      _controller = c;
      await c.startImageStream(_onFrame);
      if (mounted) setState(() => _ready = true);
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.code.startsWith('CameraAccessDenied')
            ? 'Please let me use the camera so I can see shapes.'
            : 'The camera had a problem. (${e.description ?? e.code})';
      });
    } catch (e) {
      if (mounted) setState(() => _error = 'Something went wrong. ($e)');
    }
  }

  Future<void> _releaseCamera({bool notify = true}) async {
    final c = _controller;
    _controller = null;
    if (notify && mounted) setState(() => _ready = false);
    if (c == null) return;
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
    } catch (_) {}
    await c.dispose();
  }

  // ---------------------------------------------------------------------
  // Frame handling
  // ---------------------------------------------------------------------

  void _onFrame(CameraImage image) {
    final c = _controller;
    if (c == null || _busy || _paused) return;
    final now = DateTime.now();
    if (now.difference(_lastRun).inMilliseconds < 90) return; // ~10 looks/second
    _lastRun = now;
    _busy = true;
    try {
      final r = ShapeDetector.instance.detect(image, c.description.sensorOrientation);
      _handleResult(r, now);
    } catch (e) {
      debugPrint('detect failed: $e');
    } finally {
      _busy = false;
    }
  }

  void _handleResult(DetectionResult? r, DateTime now) {
    if (!mounted) return;
    if (r == null) {
      _misses++;
      if (_misses >= 3) {
        _stableLabel = null;
        _stableCount = 0;
        if (_current != null) setState(() => _current = null);
      }
      return;
    }
    _misses = 0;
    if (r.label == _stableLabel) {
      _stableCount++;
    } else {
      _stableLabel = r.label;
      _stableCount = 1;
    }
    setState(() => _current = r);

    if (_continuous &&
        _stableCount >= 4 &&
        now.difference(_lastAutoSave).inMilliseconds > 2500) {
      _lastAutoSave = now;
      _save(r);
    }
  }

  // ---------------------------------------------------------------------
  // Buttons
  // ---------------------------------------------------------------------

  Future<void> _save(DetectionResult r) async {
    final png = await renderCapturePng(r.silhouette);
    await DbHelper.instance.insert(label: r.label, confidence: r.confidence, png: png);
    if (!mounted) return;
    setState(() => _flash = true);
    Future.delayed(const Duration(milliseconds: 150), () {
      if (mounted) setState(() => _flash = false);
    });
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(
        duration: const Duration(milliseconds: 1200),
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.ink,
        content: Text('Saved ${prettyLabel(r.label)}!',
            style: const TextStyle(fontSize: 18, color: Colors.white)),
      ));
  }

  void _shutter() {
    final r = _current;
    if (r == null) {
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(const SnackBar(
          duration: Duration(milliseconds: 1400),
          behavior: SnackBarBehavior.floating,
          content: Text("I can't see a shape yet. Try again!", style: TextStyle(fontSize: 18)),
        ));
      return;
    }
    _save(r);
  }

  Future<void> _togglePause() async {
    final c = _controller;
    if (c == null) return;
    if (_paused) {
      await c.resumePreview();
      await c.startImageStream(_onFrame);
      if (mounted) setState(() => _paused = false);
    } else {
      setState(() => _paused = true);
      if (c.value.isStreamingImages) await c.stopImageStream();
      await c.pausePreview();
    }
  }

  Future<void> _toggleContinuous() async {
    if (_paused) await _togglePause();
    if (mounted) setState(() => _continuous = !_continuous);
  }

  Future<void> _openHistory() async {
    _inHistory = true;
    await _releaseCamera();
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const HistoryScreen()),
    );
    _inHistory = false;
    if (mounted) _initCamera();
  }

  Future<void> _exit() async {
    final yes = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Close Shape Buddy?',
            style: TextStyle(fontWeight: FontWeight.w700)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Stay', style: TextStyle(fontSize: 18)),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Exit', style: TextStyle(fontSize: 18)),
          ),
        ],
      ),
    );
    if (yes == true) {
      await _releaseCamera();
      await SystemNavigator.pop();
    }
  }

  // ---------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _topBar(),
            Expanded(child: _preview()),
            _banner(),
            _controls(),
            const SizedBox(height: 14),
          ],
        ),
      ),
    );
  }

  Widget _topBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 8, 14, 4),
      child: Row(
        children: [
          _PillButton(icon: Icons.close_rounded, label: 'Exit', onTap: _exit),
          const Spacer(),
          _PillButton(
            icon: Icons.history_rounded,
            label: 'History',
            onTap: _openHistory,
          ),
        ],
      ),
    );
  }

  Widget _preview() {
    final c = _controller;
    Widget content;
    if (_error != null) {
      content = Center(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const ShapeIcon(ShapeKind.circle, size: 56, filled: false),
              const SizedBox(height: 16),
              Text(_error!, textAlign: TextAlign.center, style: const TextStyle(fontSize: 20)),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: _initCamera,
                child: const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: Text('Try again', style: TextStyle(fontSize: 18)),
                ),
              ),
            ],
          ),
        ),
      );
    } else if (!_ready || c == null || !c.value.isInitialized) {
      content = const Center(child: CircularProgressIndicator());
    } else {
      content = Center(
        child: AspectRatio(
          aspectRatio: 1 / c.value.aspectRatio,
          child: Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(c),
              if (_current != null)
                CustomPaint(painter: OutlinePainter(_current!.outline)),
              IgnorePointer(
                child: AnimatedOpacity(
                  opacity: _flash ? 0.7 : 0.0,
                  duration: const Duration(milliseconds: 120),
                  child: Container(color: Colors.white),
                ),
              ),
              if (_paused)
                Align(
                  alignment: Alignment.topCenter,
                  child: Container(
                    margin: const EdgeInsets.only(top: 12),
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppColors.ink.withValues(alpha: 0.8),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text('Paused',
                        style: TextStyle(color: Colors.white, fontSize: 16)),
                  ),
                ),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(28),
        child: Container(
          decoration: BoxDecoration(
            color: AppColors.blueSoft,
            borderRadius: BorderRadius.circular(28),
            border: Border.all(color: AppColors.blueSoft, width: 4),
          ),
          child: content,
        ),
      ),
    );
  }

  Widget _banner() {
    final r = _current;
    final kind = r == null ? null : shapeKindFromLabel(r.label);
    return SizedBox(
      height: 64,
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 180),
        child: r == null
            ? Text(
                'Show me a shape!',
                key: const ValueKey('none'),
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.w700,
                  color: AppColors.ink.withValues(alpha: 0.55),
                ),
              )
            : Row(
                key: ValueKey(r.label),
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (kind != null) ShapeIcon(kind, size: 40),
                  const SizedBox(width: 12),
                  Text('${prettyLabel(r.label)}!',
                      style: const TextStyle(fontSize: 34, fontWeight: FontWeight.w800)),
                ],
              ),
      ),
    );
  }

  Widget _controls() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        _RoundAction(
          icon: Icons.autorenew_rounded,
          label: 'Auto',
          active: _continuous,
          onTap: _toggleContinuous,
        ),
        _Shutter(onTap: _shutter),
        _RoundAction(
          icon: _paused ? Icons.play_arrow_rounded : Icons.pause_rounded,
          label: _paused ? 'Resume' : 'Pause',
          active: _paused,
          onTap: _togglePause,
        ),
      ],
    );
  }
}

// -------------------------------------------------------------------------
// Small building blocks
// -------------------------------------------------------------------------

class _PillButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _PillButton({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppColors.blueSoft,
      borderRadius: BorderRadius.circular(28),
      child: InkWell(
        borderRadius: BorderRadius.circular(28),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 26, color: AppColors.ink),
              const SizedBox(width: 6),
              Text(label,
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
            ],
          ),
        ),
      ),
    );
  }
}

class _RoundAction extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback onTap;
  const _RoundAction({
    required this.icon,
    required this.label,
    required this.onTap,
    this.active = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Material(
          color: active ? AppColors.blue : AppColors.blueSoft,
          shape: const CircleBorder(),
          child: InkWell(
            customBorder: const CircleBorder(),
            onTap: onTap,
            child: SizedBox(
              width: 68,
              height: 68,
              child: Icon(icon, size: 34, color: active ? Colors.white : AppColors.ink),
            ),
          ),
        ),
        const SizedBox(height: 6),
        Text(label, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
      ],
    );
  }
}

/// The big round shutter at the bottom: one tap saves one shape.
class _Shutter extends StatelessWidget {
  final VoidCallback onTap;
  const _Shutter({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            width: 96,
            height: 96,
            padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white,
              border: Border.all(color: AppColors.blue, width: 6),
            ),
            child: Container(
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                color: AppColors.blue,
              ),
              child: const Icon(Icons.camera_alt_rounded, size: 38, color: Colors.white),
            ),
          ),
        ),
        const SizedBox(height: 6),
        const Text('Snap!', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
      ],
    );
  }
}
