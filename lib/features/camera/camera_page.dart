import 'dart:io';
// NOTE: no 'dart:isolate' import
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart'; // <- for compute()
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../common/paths.dart';
import '../../models/exercise.dart';
import '../../models/feedback_session.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key, required this.exercise});

  final Exercise exercise;

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _cameraController;
  Future<void>? _initializeControllerFuture;
  bool _isRecording = false;
  bool _hasRecording = false;
  bool _isProcessingRecording = false; // spinner only during STOP/save
  int _repetitionCount = 0;
  Directory? _sessionDir;
  String? _recordedVideoPath;
  String? _cameraError;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    final previousController = _cameraController;
    _cameraController = null;
    await previousController?.dispose();

    setState(() {
      _cameraError = null;
    });

    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw CameraException('NoCamera', 'No cameras available on this device.');
      }

      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: true,
      );

      final initializeFuture = controller.initialize();
      setState(() {
        _cameraController = controller;
        _initializeControllerFuture = initializeFuture;
      });

      await initializeFuture;
      if (mounted) setState(() {}); // re-check _canControlRecording
    } on CameraException catch (e) {
      setState(() {
        _cameraError = e.description ?? e.code;
      });
    } catch (e) {
      setState(() {
        _cameraError = e.toString();
      });
    }
  }

  // Start recording without blocking the UI
  Future<void> _startRecording(CameraController controller) async {
    try {
      await controller.startVideoRecording();
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() => _isRecording = false);
      _showSnackBar(e.description ?? 'Camera error: ${e.code}');
    } catch (e) {
      if (!mounted) return;
      setState(() => _isRecording = false);
      _showSnackBar('Recording failed: $e');
    }
  }

  Future<void> _toggleRecording() async {
    final controller = _cameraController;
    if (controller == null) return;

    final starting = !_isRecording;
    // Show spinner only on STOP (when we copy the file)
    if (!starting) {
      setState(() => _isProcessingRecording = true);
    }

    try {
      await _initializeControllerFuture;

      if (_isRecording) {
        // ===== STOP & SAVE (heavy work off UI thread) =====
        final XFile raw = await controller.stopVideoRecording();

        final Directory activeSessionDir =
            _sessionDir ?? await Paths.makeNewSessionDir();
        final String dstPath = Paths.moviePath(activeSessionDir);
        final String srcPath = raw.path;

        // Use compute with a top-level function; pass only Strings
        final savedMoviePath = await compute(_copyRecording, {
          'src': srcPath,
          'dst': dstPath,
        });

        if (!mounted) return;
        setState(() {
          _sessionDir = activeSessionDir;
          _isRecording = false;
          _hasRecording = true;
          _recordedVideoPath = savedMoviePath;
          _repetitionCount = 12; // placeholder until analysis wires in
        });
      } else {
        // ===== START (don’t block the UI) =====
        final sessionDir = await Paths.makeNewSessionDir();
        if (!mounted) return;
        setState(() {
          _sessionDir = sessionDir;
          _hasRecording = false;
          _recordedVideoPath = null;
          _repetitionCount = 0;
          _isRecording = true;          // flip UI immediately
          _isProcessingRecording = false;
        });

        // Kick off recording in the background; if it fails, we revert flag
        // ignore: discarded_futures
        _startRecording(controller);
      }
    } on CameraException catch (e) {
      if (mounted && _isRecording && !starting) {
        setState(() => _isRecording = false);
      }
      _showSnackBar(e.description ?? 'Camera error: ${e.code}');
    } catch (e) {
      if (mounted && _isRecording && !starting) {
        setState(() => _isRecording = false);
      }
      _showSnackBar('Recording failed: $e');
    } finally {
      if (!starting && mounted) {
        setState(() => _isProcessingRecording = false);
      }
    }
  }

  void _runAnalysis() {
    final videoPath = _recordedVideoPath;
    if (videoPath == null) return;

    final session = FeedbackSession(
      exercise: widget.exercise,
      videoPath: videoPath,
      metrics: const [
        FeedbackMetric(label: 'Repetition Count', value: '12'),
        FeedbackMetric(label: 'Average Tempo', value: '2.1s'),
        FeedbackMetric(label: 'Range of Motion', value: 'Good'),
      ],
      summary:
          'Solid baseline! Keep knees aligned over toes and maintain a steady tempo.',
    );

    context.push('/feedback', extra: session);
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Widget _buildCameraPreview() {
    if (_cameraError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            'Camera unavailable\n$_cameraError',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Theme.of(context).colorScheme.onSurface,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      );
    }

    final future = _initializeControllerFuture;
    final controller = _cameraController;

    if (future == null || controller == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return FutureBuilder<void>(
      future: future,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done &&
            controller.value.isInitialized) {
          return CameraPreview(controller);
        }

        if (snapshot.hasError) {
          return Center(
            child: Text(
              'Failed to initialize the camera',
              style: TextStyle(color: Theme.of(context).colorScheme.onSurface),
            ),
          );
        }

        return const Center(child: CircularProgressIndicator());
      },
    );
  }

  bool get _canControlRecording {
    final controller = _cameraController;
    return !_isProcessingRecording &&
        _cameraError == null &&
        controller != null &&
        controller.value.isInitialized;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.exercise.name)),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    ColoredBox(color: Theme.of(context).colorScheme.surfaceVariant),
                    _buildCameraPreview(),
                    Align(
                      alignment: Alignment.topRight,
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.4),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 8,
                            ),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              crossAxisAlignment: CrossAxisAlignment.end,
                              children: [
                                Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      _isRecording
                                          ? Icons.fiber_manual_record
                                          : Icons.stop_circle_outlined,
                                      size: 14,
                                      color: _isRecording
                                          ? Colors.redAccent
                                          : Colors.white,
                                    ),
                                    const SizedBox(width: 6),
                                    Text(
                                      _isRecording
                                          ? 'Recording…'
                                          : _hasRecording
                                              ? 'Ready'
                                              : 'Idle',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'Reps: $_repetitionCount',
                                  style: const TextStyle(color: Colors.white),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                    if (_isProcessingRecording)
                      const Align(
                        alignment: Alignment.center,
                        child: CircularProgressIndicator(),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _canControlRecording ? _toggleRecording : null,
              icon: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record),
              label: Text(
                _isRecording
                    ? 'Stop Recording'
                    : _hasRecording
                        ? 'Record Again'
                        : 'Start Recording',
              ),
              style: FilledButton.styleFrom(
                backgroundColor:
                    _isRecording ? Colors.red : Theme.of(context).colorScheme.primary,
              ),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _hasRecording ? _runAnalysis : null,
              icon: const Icon(Icons.play_circle_fill),
              label: const Text('Run Analysis'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () => context.go('/'),
              icon: const Icon(Icons.arrow_back),
              label: const Text('Choose Another Exercise'),
            ),
          ],
        ),
      ),
    );
  }
}

/// Top-level function used by `compute`.
/// Copies from payload['src'] to payload['dst'] and returns the dst path.
Future<String> _copyRecording(Map<String, String> payload) async {
  final src = payload['src']!;
  final dst = payload['dst']!;

  final sourceFile = File(src);
  if (!await sourceFile.exists()) {
    throw Exception('Recording source not found at $src');
  }

  final destinationFile = File(dst);
  final destinationDir = destinationFile.parent;
  if (!await destinationDir.exists()) {
    await destinationDir.create(recursive: true);
  }

  if (await destinationFile.exists()) {
    await destinationFile.delete();
  }

  await sourceFile.copy(destinationFile.path);

  // Optional: cleanup temp file
  try {
    await sourceFile.delete();
  } catch (_) {}

  return destinationFile.path;
}
