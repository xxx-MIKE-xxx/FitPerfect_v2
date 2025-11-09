import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import 'yolo_result.dart';

class YoloOverlay extends StatefulWidget {
  const YoloOverlay({
    super.key,
    required this.controller,
    required this.result,
    this.scoreThreshold = 0.3,
  });

  final VideoPlayerController controller;
  final YoloResult result;
  final double scoreThreshold;

  @override
  State<YoloOverlay> createState() => _YoloOverlayState();
}

class _YoloOverlayState extends State<YoloOverlay> {
  Timer? _timer;
  Duration _position = Duration.zero;

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  @override
  void didUpdateWidget(covariant YoloOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      _position = widget.controller.value.position;
      _timer?.cancel();
      _startTimer();
    }
  }

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(milliseconds: 33), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (!widget.controller.value.isInitialized) {
        return;
      }
      final position = widget.controller.value.position;
      if (position != _position) {
        setState(() {
          _position = position;
        });
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.controller.value.isInitialized || widget.result.frames.isEmpty) {
      return const SizedBox.shrink();
    }

    final sampledFps = widget.result.sampledFps;
    final fallbackFps = widget.result.fps > 0 ? widget.result.fps : 30.0;
    final step = sampledFps > 0 ? (1.0 / sampledFps) : (1.0 / fallbackFps);
    final positionSeconds = _position.inMilliseconds / 1000.0;
    final frameIndex = (positionSeconds / step)
        .round()
        .clamp(0, widget.result.frames.length - 1);
    final boxes = widget.result.frames[frameIndex]
        .boxes
        .where((box) => box.score >= widget.scoreThreshold)
        .toList(growable: false);

    return IgnorePointer(
      child: CustomPaint(
        size: Size.infinite,
        painter: _YoloPainter(
          videoWidth: widget.result.videoWidth,
          videoHeight: widget.result.videoHeight,
          boxes: boxes,
        ),
      ),
    );
  }
}

class _YoloPainter extends CustomPainter {
  _YoloPainter({
    required this.videoWidth,
    required this.videoHeight,
    required this.boxes,
  });

  final int videoWidth;
  final int videoHeight;
  final List<YoloBox> boxes;

  @override
  void paint(Canvas canvas, Size size) {
    if (videoWidth <= 0 || videoHeight <= 0 || boxes.isEmpty) {
      return;
    }

    final scale = math.min(size.width / videoWidth, size.height / videoHeight);
    final dx = (size.width - videoWidth * scale) / 2;
    final dy = (size.height - videoHeight * scale) / 2;

    final rectPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (final box in boxes) {
      final left = dx + box.x * scale;
      final top  = dy + box.y * scale;
      final rect = Rect.fromLTWH(left, top, box.w * scale, box.h * scale);
      canvas.drawRect(rect, rectPaint);

      final labelText = box.label.isNotEmpty
          ? '${box.label} ${(box.score * 100).round()}%'
          : '${box.cls} ${(box.score * 100).round()}%';
      if (labelText.trim().isEmpty) {
        continue;
      }

      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final padding = const EdgeInsets.symmetric(horizontal: 6, vertical: 2);
      final labelWidth = textPainter.width + padding.horizontal;
      final labelHeight = textPainter.height + padding.vertical;
      final labelLeft = rect.left;
      final labelTop = math.max(rect.top - labelHeight, dy);
      final backgroundRect = Rect.fromLTWH(
        labelLeft,
        labelTop,
        labelWidth,
        labelHeight,
      );

      final backgroundPaint = Paint()
        ..color = Colors.greenAccent.withOpacity(0.85)
        ..style = PaintingStyle.fill;
      canvas.drawRRect(
        RRect.fromRectAndRadius(backgroundRect, const Radius.circular(4)),
        backgroundPaint,
      );

      textPainter.paint(
        canvas,
        Offset(
          backgroundRect.left + padding.left,
          backgroundRect.top + padding.top,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant _YoloPainter oldDelegate) {
    return oldDelegate.videoWidth != videoWidth ||
        oldDelegate.videoHeight != videoHeight ||
        !listEquals(oldDelegate.boxes, boxes);
  }
}
