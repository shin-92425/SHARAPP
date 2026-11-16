import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import '../models/capture_record.dart';

/// SQLite storage for the history page. Each row keeps the shape name,
/// how sure the app was, the path of the saved picture and the date/time.
class DbHelper {
  DbHelper._();
  static final DbHelper instance = DbHelper._();

  Database? _db;

  Future<Database> get _database async {
    if (_db != null) return _db!;
    final dir = await getDatabasesPath();
    _db = await openDatabase(
      p.join(dir, 'shape_buddy.db'),
      version: 1,
      onCreate: (db, v) async {
        await db.execute('''
          CREATE TABLE captures(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT NOT NULL
          )
        ''');
      },
    );
    return _db!;
  }

  Future<Directory> _imageDir() async {
    final base = await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(base.path, 'captures'));
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  /// Saves the picture to disk and the details to the database.
  Future<CaptureRecord> insert({
    required String label,
    required double confidence,
    required Uint8List png,
  }) async {
    final now = DateTime.now();
    final dir = await _imageDir();
    final file = File(p.join(dir.path, '${now.microsecondsSinceEpoch}.png'));
    await file.writeAsBytes(png, flush: true);

    final db = await _database;
    final record = CaptureRecord(
      label: label,
      confidence: confidence,
      imagePath: file.path,
      createdAt: now,
    );
    final id = await db.insert('captures', record.toMap());
    return CaptureRecord(
      id: id,
      label: label,
      confidence: confidence,
      imagePath: file.path,
      createdAt: now,
    );
  }

  Future<List<CaptureRecord>> all() async {
    final db = await _database;
    final rows = await db.query('captures', orderBy: 'created_at DESC');
    return rows.map(CaptureRecord.fromMap).toList();
  }

  Future<void> delete(CaptureRecord r) async {
    final db = await _database;
    await db.delete('captures', where: 'id = ?', whereArgs: [r.id]);
    await _deleteFile(r.imagePath);
  }

  Future<void> deleteAll() async {
    final db = await _database;
    final rows = await db.query('captures', columns: ['image_path']);
    await db.delete('captures');
    for (final row in rows) {
      await _deleteFile(row['image_path'] as String);
    }
  }

  Future<void> _deleteFile(String path) async {
    try {
      final f = File(path);
      if (await f.exists()) await f.delete();
    } catch (_) {
      // a missing picture should never break the history page
    }
  }
}
