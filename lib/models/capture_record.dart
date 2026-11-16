class CaptureRecord {
  final int? id;
  final String label;
  final double confidence;
  final String imagePath;
  final DateTime createdAt;

  const CaptureRecord({
    this.id,
    required this.label,
    required this.confidence,
    required this.imagePath,
    required this.createdAt,
  });

  Map<String, Object?> toMap() => {
        if (id != null) 'id': id,
        'label': label,
        'confidence': confidence,
        'image_path': imagePath,
        'created_at': createdAt.toIso8601String(),
      };

  factory CaptureRecord.fromMap(Map<String, Object?> m) => CaptureRecord(
        id: m['id'] as int?,
        label: m['label'] as String,
        confidence: (m['confidence'] as num).toDouble(),
        imagePath: m['image_path'] as String,
        createdAt: DateTime.parse(m['created_at'] as String),
      );
}
