import 'dart:io';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:video_player/video_player.dart';

import '../../models/feedback_session.dart';
import '../../yolo/yolo_overlay.dart';
import '../../yolo/yolo_result.dart';

class FeedbackPage extends StatefulWidget {
  const FeedbackPage({super.key, required this.session});

  final FeedbackSession session;

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  late final VideoPlayerController _videoController;
  late final Future<void> _initializeVideoFuture;
  YoloResult? _yolo;

  @override
  void initState() {
    super.initState();
    _videoController = VideoPlayerController.file(File(widget.session.videoPath));
    _initializeVideoFuture = _videoController.initialize().then((_) async {
      _videoController
        ..setLooping(true)
        ..setVolume(1.0);

      final videoFile = File(widget.session.videoPath);
      final jsonPath = '${videoFile.parent.path}/yolo_detections.json';
      final result = await YoloResult.load(jsonPath);
      if (!mounted) {
        return;
      }

      setState(() {
        _yolo = result;
      });
    });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  void _togglePlayback() {
    if (!_videoController.value.isInitialized) {
      return;
    }

    if (_videoController.value.isPlaying) {
      _videoController.pause();
    } else {
      _videoController.play();
    }

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final session = widget.session;

    return Scaffold(
      appBar: AppBar(title: Text('${session.exercise.name} Feedback')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: AspectRatio(
              aspectRatio: _videoController.value.isInitialized
                  ? _videoController.value.aspectRatio
                  : 16 / 9,
              child: ColoredBox(
                color: Theme.of(context).colorScheme.surfaceVariant,
                child: FutureBuilder<void>(
                  future: _initializeVideoFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Center(child: CircularProgressIndicator());
                    }

                    if (snapshot.hasError || !_videoController.value.isInitialized) {
                      final errorMessage = snapshot.error?.toString() ??
                          'Unable to load recorded video.';
                      return Center(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Text(
                            errorMessage,
                            textAlign: TextAlign.center,
                          ),
                        ),
                      );
                    }

                    return Stack(
                      alignment: Alignment.center,
                      children: [
                        VideoPlayer(_videoController),
                        if (_yolo != null)
                          YoloOverlay(
                            controller: _videoController,
                            result: _yolo!,
                          ),
                        if (!_videoController.value.isPlaying)
                          IconButton(
                            iconSize: 72,
                            color: Colors.white.withOpacity(0.9),
                            icon: const Icon(Icons.play_circle_fill),
                            onPressed: _togglePlayback,
                          ),
                        Positioned(
                          bottom: 12,
                          left: 12,
                          right: 12,
                          child: VideoProgressIndicator(
                            _videoController,
                            allowScrubbing: true,
                            colors: VideoProgressColors(
                              playedColor: Theme.of(context).colorScheme.primary,
                              bufferedColor: Theme.of(context).colorScheme.surfaceTint,
                              backgroundColor:
                                  Theme.of(context).colorScheme.onSurface.withOpacity(0.2),
                            ),
                          ),
                        ),
                        Positioned(
                          bottom: 12,
                          right: 12,
                          child: FilledButton.tonalIcon(
                            onPressed: _togglePlayback,
                            icon: Icon(
                              _videoController.value.isPlaying
                                  ? Icons.pause
                                  : Icons.play_arrow,
                            ),
                            label: Text(
                              _videoController.value.isPlaying ? 'Pause' : 'Play',
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Card(
            child: ListTile(
              leading: const Icon(Icons.save_alt_outlined),
              title: const Text('Session Video'),
              subtitle: Text(session.videoPath),
            ),
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Session Summary',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Text(session.summary),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Key Metrics',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 12),
                  ...session.metrics.map(
                    (metric) => Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      child: Row(
                        children: [
                          Icon(
                            Icons.analytics_outlined,
                            color: Theme.of(context).colorScheme.primary,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  metric.label,
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodyLarge
                                      ?.copyWith(fontWeight: FontWeight.w600),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  metric.value,
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodyMedium
                                      ?.copyWith(color: Theme.of(context).hintColor),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => context.pop(),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Record Again'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => context.go('/'),
                  icon: const Icon(Icons.home_outlined),
                  label: const Text('Back to Exercises'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
