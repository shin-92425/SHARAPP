import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

import '../widgets/shape_painters.dart';

/// Speaks shape names out loud (English only).
class SpeechService {
  SpeechService._();
  static final SpeechService instance = SpeechService._();

  final FlutterTts _tts = FlutterTts();
  bool _ready = false;

  Future<void> _init() async {
    if (_ready) return;
    try {
      await _tts.setLanguage('en-US');
      await _tts.setSpeechRate(0.45); // a little slower, kid friendly
      await _tts.setPitch(1.1);
      await _tts.setVolume(1.0);
      await _tts.awaitSpeakCompletion(false);
      _ready = true;
    } catch (e) {
      debugPrint('TTS init failed: $e');
    }
  }

  /// Say the shape name, e.g. "Circle". Interrupts anything already speaking.
  Future<void> speakShape(String label) async {
    if (label.isEmpty) return;
    await _init();
    try {
      await _tts.stop();
      await _tts.speak(prettyLabel(label));
    } catch (e) {
      debugPrint('TTS speak failed: $e');
    }
  }

  Future<void> stop() async {
    try {
      await _tts.stop();
    } catch (_) {}
  }
}
