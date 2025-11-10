import 'dart:io';

import 'package:archive/archive.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:path/path.dart' as p;
import 'package:share_plus/share_plus.dart';
import 'package:video_player/video_player.dart';
import 'dart:typed_data';
import '../../models/feedback_session.dart';
import '../../rtmpose/rtmpose_overlay.dart';
import '../../rtmpose/rtmpose_result.dart';
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
  RtmposeResult? _rtmpose;
  bool _showPose = true;
  bool _isSharing = false;

  @override
  void initState() {
    super.initState();
    _videoController = VideoPlayerController.file(File(widget.session.videoPath));
    _initializeVideoFuture = _videoController.initialize().then((_) async {
      _videoController
        ..setLooping(true)
        ..setVolume(1.0);

      final session = widget.session;
      final sessionDir = Directory(session.sessionDir);
      final jsonPath = session.yoloJsonPath ??
          p.join(sessionDir.path, 'yolo_detections.json');
      final result = await YoloResult.load(jsonPath);

      final poseJsonPath = session.poseJsonPath;
      RtmposeResult? poseResult;
      if (poseJsonPath != null) {
        poseResult = await RtmposeResult.load(poseJsonPath);
      }

      final fallbackPosePath = p.join(sessionDir.path, 'rtmpose_keypoints.json');
      if (poseResult == null && fallbackPosePath != poseJsonPath) {
        poseResult = await RtmposeResult.load(fallbackPosePath);
      }
      if (!mounted) {
        return;
      }

      setState(() {
        _yolo = result;
        _rtmpose = poseResult;
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

  Future<void> _shareRun() async {
    if (_isSharing) {
      return;
    }

    setState(() {
      _isSharing = true;
    });

    try {
      final zipPath = await zipRun(widget.session);
      await Share.shareXFiles(
        [XFile(zipPath)],
        text: 'FitPerfect run export',
      );
    } catch (error) {
      if (!mounted) {
        return;
      }
      final messenger = ScaffoldMessenger.of(context);
      messenger.showSnackBar(
        SnackBar(content: Text('Unable to share run: $error')),
      );
    } finally {
      if (!mounted) {
        return;
      }
      setState(() {
        _isSharing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final session = widget.session;

    return Scaffold(
      appBar: AppBar(
        title: Text('${session.exercise.name} Feedback'),
        actions: [
          if (_isSharing)
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: SizedBox(
                height: 20,
                width: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            IconButton(
              icon: const Icon(Icons.ios_share),
              tooltip: 'Share Run',
              onPressed: _shareRun,
            ),
        ],
      ),
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
                        if (_showPose && _rtmpose != null)
                          RtmposeOverlay(
                            controller: _videoController,
                            result: _rtmpose!,
                          ),
                        if (_rtmpose != null)
                          Positioned(
                            top: 12,
                            left: 12,
                            child: FilledButton.tonalIcon(
                              onPressed: () {
                                setState(() {
                                  _showPose = !_showPose;
                                });
                              },
                              icon: Icon(
                                _showPose
                                    ? Icons.visibility
                                    : Icons.visibility_off,
                              ),
                              label: Text(_showPose ? 'Pose On' : 'Pose Off'),
                            ),
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

Future<String> zipRun(FeedbackSession session) async {
  final sessionDir = Directory(session.sessionDir);
  if (!await sessionDir.exists()) {
    throw StateError('Session directory not found: ${session.sessionDir}');
  }

  final archive = Archive();
  final seen = <String>{};
  final rootName = p.basename(sessionDir.path).isEmpty
      ? 'session'
      : p.basename(sessionDir.path);

  Future<void> addDirectory(Directory directory, String relativePath) async {
    if (relativePath.isNotEmpty) {
      final dirPath = relativePath.endsWith('/') ? relativePath : '$relativePath/';
      archive.addFile(ArchiveFile(dirPath, 0, Uint8List(0))); // directory entry
    }

    await for (final entity
        in directory.list(recursive: false, followLinks: false)) {
      final name = p.basename(entity.path);
      final relativeEntry =
          relativePath.isEmpty ? name : p.posix.join(relativePath, name);

      if (entity is File) {
        final normalized = p.normalize(entity.path);
        if (!seen.add(normalized)) {
          continue;
        }

        final bytes = await entity.readAsBytes();
        archive.addFile(ArchiveFile(relativeEntry, bytes.length, bytes));
      } else if (entity is Directory) {
        await addDirectory(entity, relativeEntry);
      }
    }
  }

  await addDirectory(sessionDir, rootName);

  final additionalPaths = <String?>{
    session.videoPath,
    session.yoloJsonPath,
    session.poseJsonPath,
    session.posePreviewPath,
    session.motionJsonPath,
    session.motionPreviewPath,
  }..removeWhere((path) => path == null || path!.isEmpty);

  for (final path in additionalPaths.cast<String>()) {
    final file = File(path);
    if (!await file.exists()) {
      continue;
    }

    final normalized = p.normalize(file.path);
    if (seen.contains(normalized)) {
      continue;
    }

    if (p.isWithin(sessionDir.path, file.path)) {
      continue;
    }

    final bytes = await file.readAsBytes();
    final baseName = p.basename(file.path);
    String entryName = p.posix.join('external', baseName);
    if (archive.files.any((entry) => entry.name == entryName)) {
      final nameWithoutExtension = p.basenameWithoutExtension(baseName);
      final extension = p.extension(baseName);
      var counter = 1;
      while (archive.files.any((entry) => entry.name == entryName)) {
        final numberedName = '${nameWithoutExtension}_$counter$extension';
        entryName = p.posix.join('external', numberedName);
        counter++;
      }
    }

    archive.addFile(ArchiveFile(entryName, bytes.length, bytes));
    seen.add(normalized);
  }

  final encoder = ZipEncoder();
  final data = encoder.encode(archive);
  if (data == null) {
    throw StateError('Failed to encode archive');
  }

  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final zipName = 'fitperfect_run_$timestamp.zip';
  final outputDir = sessionDir.parent;
  final outputPath = p.join(outputDir.path, zipName);
  final outputFile = File(outputPath);

  if (await outputFile.exists()) {
    await outputFile.delete();
  }

  await outputFile.writeAsBytes(data, flush: true);
  return outputPath;
}
