import 'package:flutter/material.dart';

/// Only three colours are used across the whole app:
/// cream (background), ink (text) and one friendly blue (everything active).
class AppColors {
  static const cream = Color(0xFFFFF8EE);
  static const ink = Color(0xFF3B3F51);
  static const blue = Color(0xFF4F9DA6);
  static const blueSoft = Color(0xFFDCEEF0); // a pale tint of the blue
}

ThemeData buildTheme() {
  final base = ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AppColors.blue,
      brightness: Brightness.light,
    ).copyWith(
      primary: AppColors.blue,
      onPrimary: Colors.white,
      surface: AppColors.cream,
      onSurface: AppColors.ink,
    ),
  );
  return base.copyWith(
    scaffoldBackgroundColor: AppColors.cream,
    textTheme: base.textTheme.apply(
      bodyColor: AppColors.ink,
      displayColor: AppColors.ink,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.cream,
      foregroundColor: AppColors.ink,
      elevation: 0,
      centerTitle: true,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: AppColors.cream,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
    ),
  );
}
