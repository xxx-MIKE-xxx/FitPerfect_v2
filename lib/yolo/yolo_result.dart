import 'dart:convert';
import 'dart:io';

class YoloResult {
  const YoloResult({
    required this.videoWidth,
    required this.videoHeight,
    required this.fps,
    required this.sampledFps,
    required this.frames,
  });

  final int videoWidth;
  final int videoHeight;
  final double fps;
  final double sampledFps;
  final List<YoloFrame> frames;

  static Future<YoloResult?> load(String jsonPath) async {
    final file = File(jsonPath);
    if (!await file.exists()) {
      return null;
    }

    try {
      final raw = await file.readAsString();
      final data = jsonDecode(raw) as Map<String, dynamic>;
      final framesJson = data['frames'] as List<dynamic>? ?? const [];

      return YoloResult(
        videoWidth: (data['videoWidth'] as num?)?.toInt() ?? 0,
        videoHeight: (data['videoHeight'] as num?)?.toInt() ?? 0,
        fps: (data['fps'] as num?)?.toDouble() ?? 0,
        sampledFps: (data['sampledFps'] as num?)?.toDouble() ?? 0,
        frames: framesJson
            .map((frame) => YoloFrame.fromJson(frame as Map<String, dynamic>))
            .toList(growable: false),
      );
    } catch (_) {
      return null;
    }
  }
}

class YoloFrame {
  const YoloFrame({required this.t, required this.boxes});

  factory YoloFrame.fromJson(Map<String, dynamic> json) {
    final boxesJson = json['boxes'] as List<dynamic>? ?? const [];
    return YoloFrame(
      t: (json['t'] as num?)?.toDouble() ?? 0,
      boxes: boxesJson
          .map((box) => YoloBox.fromJson(box as Map<String, dynamic>))
          .toList(growable: false),
    );
  }

  final double t;
  final List<YoloBox> boxes;
}

class YoloBox {
  const YoloBox({
    required this.cls,
    required this.label,
    required this.score,
    required this.x,
    required this.y,
    required this.w,
    required this.h,
  });

  factory YoloBox.fromJson(Map<String, dynamic> json) {
    return YoloBox(
      cls: (json['cls'] as num?)?.toInt() ?? 0,
      label: json['label'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0,
      x: (json['x'] as num?)?.toDouble() ?? 0,
      y: (json['y'] as num?)?.toDouble() ?? 0,
      w: (json['w'] as num?)?.toDouble() ?? 0,
      h: (json['h'] as num?)?.toDouble() ?? 0,
    );
  }

  final int cls;
  final String label;
  final double score;
  final double x;
  final double y;
  final double w;
  final double h;
}
