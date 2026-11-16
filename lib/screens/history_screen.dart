import 'dart:io';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/capture_record.dart';
import '../services/db_helper.dart';
import '../theme.dart';
import '../widgets/shape_painters.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<CaptureRecord>? _items;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final rows = await DbHelper.instance.all();
    if (mounted) setState(() => _items = rows);
  }

  Future<bool> _confirm(String title, String yes) async {
    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Keep', style: TextStyle(fontSize: 18)),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(yes, style: const TextStyle(fontSize: 18)),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  Future<void> _delete(CaptureRecord r) async {
    if (!await _confirm('Delete this shape?', 'Delete')) return;
    await DbHelper.instance.delete(r);
    await _load();
  }

  Future<void> _deleteAll() async {
    if (!await _confirm('Delete all shapes?', 'Delete all')) return;
    await DbHelper.instance.deleteAll();
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final items = _items;
    return Scaffold(
      appBar: AppBar(
        title: const Text('My Shapes',
            style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800)),
        actions: [
          if (items != null && items.isNotEmpty)
            IconButton(
              tooltip: 'Delete all',
              iconSize: 28,
              onPressed: _deleteAll,
              icon: const Icon(Icons.delete_sweep_rounded),
            ),
        ],
      ),
      body: items == null
          ? const Center(child: CircularProgressIndicator())
          : items.isEmpty
              ? _empty()
              : ListView.separated(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                  itemCount: items.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (_, i) => _card(items[i]),
                ),
    );
  }

  Widget _card(CaptureRecord r) {
    final when = DateFormat('MMM d, y  •  h:mm a').format(r.createdAt);
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.blueSoft, width: 3),
      ),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.file(
              File(r.imagePath),
              width: 84,
              height: 84,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => Container(
                width: 84,
                height: 84,
                color: AppColors.blueSoft,
                alignment: Alignment.center,
                child: ShapeIcon(shapeKindFromLabel(r.label) ?? ShapeKind.circle, size: 44),
              ),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(prettyLabel(r.label),
                    style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w800)),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Icon(Icons.access_time_rounded,
                        size: 16, color: AppColors.ink.withValues(alpha: 0.6)),
                    const SizedBox(width: 4),
                    Flexible(
                      child: Text(
                        when,
                        style: TextStyle(
                          fontSize: 14,
                          color: AppColors.ink.withValues(alpha: 0.7),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          IconButton(
            tooltip: 'Delete',
            iconSize: 30,
            color: AppColors.blue,
            onPressed: () => _delete(r),
            icon: const Icon(Icons.delete_outline_rounded),
          ),
        ],
      ),
    );
  }

  Widget _empty() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ShapeIcon(ShapeKind.circle, size: 44, filled: false),
                SizedBox(width: 12),
                ShapeIcon(ShapeKind.star, size: 44, filled: false),
                SizedBox(width: 12),
                ShapeIcon(ShapeKind.heart, size: 44, filled: false),
              ],
            ),
            const SizedBox(height: 22),
            const Text('No shapes yet',
                style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            Text(
              'Take a picture of a shape and it will show up here.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 18, color: AppColors.ink.withValues(alpha: 0.7)),
            ),
          ],
        ),
      ),
    );
  }
}
