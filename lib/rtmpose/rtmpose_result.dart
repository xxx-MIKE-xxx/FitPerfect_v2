import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

class RtmposeResult {
  const RtmposeResult({
    required this.ok,
    required this.fps,
    required this.sampledFps,
    required this.videoWidth,
    required this.videoHeight,
    required this.frames,
    required this.usesNormalizedCoordinates,
  });

  final bool ok;
  final double fps;
  final double sampledFps;
  final int videoWidth;
  final int videoHeight;
  final List<RtmposeFrame> frames;
  final bool usesNormalizedCoordinates;

  static Future<RtmposeResult?> load(String jsonPath) async {
    final file = File(jsonPath);
    if (!await file.exists()) {
      return null;
    }

    try {
      final raw = await file.readAsString();
      final json = jsonDecode(raw);
      if (json is! Map<String, dynamic>) {
        return null;
      }

      double _readDouble(dynamic value) {
        if (value is num) return value.toDouble();
        if (value is String) return double.tryParse(value) ?? 0;
        return 0;
      }

      int _readInt(dynamic value) {
        if (value is int) return value;
        if (value is num) return value.toInt();
        if (value is String) return int.tryParse(value) ?? 0;
        return 0;
      }

      final ok = json['ok'] == null ? true : json['ok'] == true;
      final fps = _readDouble(json['fps']);
      final sampledFps = _readDouble(json['sampled_fps'] ?? json['sampledFps']);
      final width = _readInt(
        json['video_width'] ??
            json['videoWidth'] ??
            (json['video_size'] is List && (json['video_size'] as List).length > 0
                ? (json['video_size'] as List)[0]
                : 0),
      );
      final height = _readInt(
        json['video_height'] ??
            json['videoHeight'] ??
            (json['video_size'] is List && (json['video_size'] as List).length > 1
                ? (json['video_size'] as List)[1]
                : 0),
      );

      final framesJson = json['frames'];
      if (framesJson is! List) {
        return null;
      }

      final frames = <RtmposeFrame>[];
      for (var i = 0; i < framesJson.length; i++) {
        final entry = framesJson[i];
        final frame = RtmposeFrame.fromJson(entry, index: i);
        if (frame != null) {
          frames.add(frame);
        }
      }

      final usesNormalized = _inferNormalizedCoordinates(frames, width, height);

      return RtmposeResult(
        ok: ok,
        fps: fps,
        sampledFps: sampledFps,
        videoWidth: width,
        videoHeight: height,
        frames: frames,
        usesNormalizedCoordinates: usesNormalized,
      );
    } catch (error, stackTrace) {
      debugPrint('Failed to load RTMPose result: $error\n$stackTrace');
      return null;
    }
  }

  static bool _inferNormalizedCoordinates(
    List<RtmposeFrame> frames,
    int videoWidth,
    int videoHeight,
  ) {
    if (frames.isEmpty) {
      return false;
    }

    if (videoWidth <= 0 || videoHeight <= 0) {
      for (final frame in frames) {
        for (final joint in frame.joints) {
          if (joint.x > 1.5 || joint.y > 1.5) {
            return false;
          }
        }
      }
      return true;
    }

    for (final frame in frames) {
      for (final joint in frame.joints) {
        if (joint.x > 1.5 || joint.y > 1.5) {
          return false;
        }
      }
    }

    return true;
  }
}

class RtmposeFrame {
  const RtmposeFrame({
    required this.time,
    required this.joints,
  });

  final double time;
  final List<RtmposeJoint> joints;

  int get visibleJointCount => joints.where((joint) => joint.isVisible).length;

  static RtmposeFrame? fromJson(dynamic json, {required int index}) {
    if (json is! Map<String, dynamic>) {
      return null;
    }

    double _readDouble(dynamic value) {
      if (value is num) return value.toDouble();
      if (value is String) return double.tryParse(value) ?? 0;
      return 0;
    }

    final time = json.containsKey('t')
        ? _readDouble(json['t'])
        : json.containsKey('time')
            ? _readDouble(json['time'])
            : index.toDouble();

    final keypoints = json['keypoints'] ?? json['joints'] ?? const [];
    if (keypoints is! List || keypoints.isEmpty) {
      return RtmposeFrame(time: time, joints: const []);
    }

    final joints = <RtmposeJoint>[];
    final flatNumbers = keypoints.every((element) => element is num);

    if (flatNumbers) {
      for (var i = 0; i + 1 < keypoints.length; i += 3) {
        final x = _readDouble(keypoints[i]);
        final y = _readDouble(keypoints[i + 1]);
        final score = i + 2 < keypoints.length ? _readDouble(keypoints[i + 2]) : 0.0;
        joints.add(
          RtmposeJoint(
            index: joints.length,
            x: x,
            y: y,
            score: score,
            isVisible: score > 0,
          ),
        );
      }
    } else {
      for (var i = 0; i < keypoints.length; i++) {
        final joint = RtmposeJoint.fromJson(keypoints[i], index: i);
        if (joint != null) {
          joints.add(joint);
        }
      }
    }

    return RtmposeFrame(time: time, joints: joints);
  }
}

class RtmposeJoint {
  const RtmposeJoint({
    required this.index,
    required this.x,
    required this.y,
    required this.score,
    required this.isVisible,
  });

  final int index;
  final double x;
  final double y;
  final double score;
  final bool isVisible;

  static RtmposeJoint? fromJson(dynamic json, {required int index}) {
    double _readDouble(dynamic value) {
      if (value is num) return value.toDouble();
      if (value is String) return double.tryParse(value) ?? 0;
      return 0;
    }

    bool _readBool(dynamic value) {
      if (value is bool) return value;
      if (value is num) return value != 0;
      if (value is String) {
        final lower = value.toLowerCase();
        return lower == 'true' || lower == '1' || lower == 'yes';
      }
      return false;
    }

    if (json is Map<String, dynamic>) {
      final x = _readDouble(json['x']);
      final y = _readDouble(json['y']);
      if (x == 0 && y == 0 && !json.containsKey('x') && !json.containsKey('y')) {
        return null;
      }
      final score = json.containsKey('score') ? _readDouble(json['score']) : 0.0;
      final visible = json.containsKey('visible')
          ? _readBool(json['visible'])
          : (score > 0);
      return RtmposeJoint(
        index: index,
        x: x,
        y: y,
        score: score,
        isVisible: visible,
      );
    }

    if (json is List && json.length >= 2) {
      final x = _readDouble(json[0]);
      final y = _readDouble(json[1]);
      final score = json.length >= 3 ? _readDouble(json[2]) : 0.0;
      return RtmposeJoint(
        index: index,
        x: x,
        y: y,
        score: score,
        isVisible: score > 0,
      );
    }

    return null;
  }
}
