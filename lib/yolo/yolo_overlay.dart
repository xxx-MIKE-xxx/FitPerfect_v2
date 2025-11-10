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

    final Size videoSize = Size(videoWidth.toDouble(), videoHeight.toDouble());
    final FittedSizes fitted = applyBoxFit(BoxFit.contain, videoSize, size);
    final Size destination = fitted.destination;
    final Offset padding = Offset(
      (size.width - destination.width) / 2,
      (size.height - destination.height) / 2,
    );

    final double sx = destination.width / videoSize.width;
    final double sy = destination.height / videoSize.height;

    Offset mapVideoToCanvas(Offset p) => Offset(
          p.dx * sx + padding.dx,
          p.dy * sy + padding.dy,
        );

    final rectPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (final box in boxes) {
      final Offset topLeft = mapVideoToCanvas(Offset(box.x.toDouble(), box.y.toDouble()));
      final Rect rect = Rect.fromLTWH(
        topLeft.dx,
        topLeft.dy,
        box.w * sx,
        box.h * sy,
      );
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

      final labelPadding = const EdgeInsets.symmetric(horizontal: 6, vertical: 2);
      final labelWidth = textPainter.width + labelPadding.horizontal;
      final labelHeight = textPainter.height + labelPadding.vertical;
      final labelLeft = rect.left;
      final labelTop = math.max(rect.top - labelHeight, padding.dy);
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
          backgroundRect.left + labelPadding.left,
          backgroundRect.top + labelPadding.top,
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
