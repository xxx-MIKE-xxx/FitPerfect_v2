import 'package:flutter/foundation.dart';

import 'exercise.dart';

@immutable
class FeedbackMetric {
  final String label;
  final String value;

  const FeedbackMetric({
    required this.label,
    required this.value,
  });
}

@immutable
class FeedbackSession {
  final Exercise exercise;
  final String videoPath;
  final String sessionDir;
  final List<FeedbackMetric> metrics;
  final String summary;
  final String? poseJsonPath;
  final String? yoloJsonPath;
  final String? posePreviewPath;

  const FeedbackSession({
    required this.exercise,
    required this.videoPath,
    required this.sessionDir,
    required this.metrics,
    required this.summary,
    this.poseJsonPath,
    this.yoloJsonPath,
    this.posePreviewPath,
  });
}
