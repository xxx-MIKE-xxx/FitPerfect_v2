import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import 'rtmpose_result.dart';

class RtmposeOverlay extends StatefulWidget {
  const RtmposeOverlay({
    super.key,
    required this.controller,
    required this.result,
    this.scoreThreshold = 0.1,
    this.showStatusChip = true,
  });

  final VideoPlayerController controller;
  final RtmposeResult result;
  final double scoreThreshold;
  final bool showStatusChip;

  @override
  State<RtmposeOverlay> createState() => _RtmposeOverlayState();
}

class _RtmposeOverlayState extends State<RtmposeOverlay> {
  Timer? _timer;
  Duration _position = Duration.zero;

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  @override
  void didUpdateWidget(covariant RtmposeOverlay oldWidget) {
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
    final frame = widget.result.frames[frameIndex];
    final joints = frame.joints
        .where(
          (joint) => joint.isVisible &&
              (joint.score <= 0 ? true : joint.score >= widget.scoreThreshold),
        )
        .toList(growable: false);

    return IgnorePointer(
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: _RtmposePainter(
                videoWidth: widget.result.videoWidth,
                videoHeight: widget.result.videoHeight,
                usesNormalizedCoordinates: widget.result.usesNormalizedCoordinates,
                joints: joints,
              ),
            ),
          ),
          if (widget.showStatusChip)
            Positioned(
              top: 12,
              right: 12,
              child: _StatusChip(
                frameIndex: frameIndex,
                totalFrames: widget.result.frames.length,
                ok: widget.result.ok,
                visibleJoints: joints.length,
              ),
            ),
        ],
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({
    required this.frameIndex,
    required this.totalFrames,
    required this.ok,
    required this.visibleJoints,
  });

  final int frameIndex;
  final int totalFrames;
  final bool ok;
  final int visibleJoints;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final bool hasJoints = visibleJoints > 0;
    final backgroundColor = hasJoints
        ? colorScheme.secondary.withOpacity(0.85)
        : Colors.orangeAccent.withOpacity(0.85);
    final textColor = hasJoints
        ? colorScheme.onSecondary
        : Colors.black87;

    return DecoratedBox(
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        child: Text(
          'Pose ${frameIndex + 1}/$totalFrames · ${ok ? 'ok' : 'err'} · joints: $visibleJoints',
          style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: textColor,
                fontWeight: FontWeight.w600,
              ),
        ),
      ),
    );
  }
}

class _RtmposePainter extends CustomPainter {
  _RtmposePainter({
    required this.videoWidth,
    required this.videoHeight,
    required this.usesNormalizedCoordinates,
    required this.joints,
  });

  final int videoWidth;
  final int videoHeight;
  final bool usesNormalizedCoordinates;
  final List<RtmposeJoint> joints;

  static const List<List<int>> _skeleton = [
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 6],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [0, 5],
    [0, 6],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    if (joints.isEmpty) {
      return;
    }

    final double resolvedWidth = videoWidth > 0
        ? videoWidth.toDouble()
        : (usesNormalizedCoordinates ? 1.0 : size.width);
    final double resolvedHeight = videoHeight > 0
        ? videoHeight.toDouble()
        : (usesNormalizedCoordinates ? 1.0 : size.height);
    if (resolvedWidth <= 0 || resolvedHeight <= 0) {
      return;
    }

    final Size videoSize = Size(resolvedWidth, resolvedHeight);
    final FittedSizes fitted = applyBoxFit(BoxFit.cover, videoSize, size);
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

    Offset resolveJointPosition(RtmposeJoint joint) {
      double px = joint.x;
      double py = joint.y;
      if (usesNormalizedCoordinates) {
        px *= videoSize.width;
        py *= videoSize.height;
      }
      return mapVideoToCanvas(Offset(px, py));
    }

    final jointPaint = Paint()
      ..color = Colors.lightBlueAccent
      ..style = PaintingStyle.fill;

    final linePaint = Paint()
      ..color = Colors.lightBlueAccent
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final jointsByIndex = {for (final joint in joints) joint.index: joint};

    for (final pair in _skeleton) {
      final first = jointsByIndex[pair[0]];
      final second = jointsByIndex[pair[1]];
      if (first == null || second == null) {
        continue;
      }
      final p1 = resolveJointPosition(first);
      final p2 = resolveJointPosition(second);
      canvas.drawLine(p1, p2, linePaint);
    }

    for (final joint in joints) {
      final point = resolveJointPosition(joint);
      canvas.drawCircle(point, 5, jointPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _RtmposePainter oldDelegate) {
    return oldDelegate.videoWidth != videoWidth ||
        oldDelegate.videoHeight != videoHeight ||
        oldDelegate.usesNormalizedCoordinates != usesNormalizedCoordinates ||
        !listEquals(oldDelegate.joints, joints);
  }
}
