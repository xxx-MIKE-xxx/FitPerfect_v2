import 'dart:io';
// NOTE: no 'dart:isolate' import
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart'; // <- for compute()
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
  static const MethodChannel _analysisChannel = MethodChannel('fitperfect/analysis');
  static const double _sampledFps = 5.0;

  CameraController? _cameraController;
  Future<void>? _initializeControllerFuture;
  bool _isRecording = false;
  bool _hasRecording = false;
  bool _isProcessingRecording = false; // spinner only during STOP/save
  int _repetitionCount = 0;
  Directory? _sessionDir;
  String? _recordedVideoPath;
  String? _yoloJsonPath;
  int _framesAnalyzed = 0;
  int _detections = 0;
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

    bool analysisDialogVisible = false;

    try {
      await _initializeControllerFuture;

      if (_isRecording) {
        if (mounted) {
          analysisDialogVisible = true;
          // ignore: discarded_futures
          showDialog<void>(
            context: context,
            barrierDismissible: false,
            builder: (_) => const _AnalyzingDialog(),
          );
        }
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

        _YoloSummary? summary;
        try {
          summary = await _runNativeYoloAnalysis(
            videoPath: savedMoviePath,
            sessionDir: activeSessionDir,
          );
        } on PlatformException catch (e) {
          _showSnackBar('YOLO failed: ${e.message ?? e.code}');
        } catch (e) {
          _showSnackBar('YOLO failed: $e');
        }

        if (!mounted) return;
        setState(() {
          _sessionDir = activeSessionDir;
          _isRecording = false;
          _hasRecording = true;
          _recordedVideoPath = savedMoviePath;
          _yoloJsonPath = summary?.jsonPath;
          _framesAnalyzed = summary?.framesProcessed ?? 0;
          _detections = summary?.detections ?? 0;
          _repetitionCount = summary?.detections ?? 0;
        });

        if (summary != null) {
          final message =
              'Analyzed ${summary.framesProcessed} frames, ${summary.detections} detections. JSON saved to ${summary.jsonPath}';
          _showSnackBar(message);
        }
      } else {
        // ===== START (don’t block the UI) =====
        final sessionDir = await Paths.makeNewSessionDir();
        if (!mounted) return;
        setState(() {
          _sessionDir = sessionDir;
          _hasRecording = false;
          _recordedVideoPath = null;
          _repetitionCount = 0;
          _isRecording = true; // flip UI immediately
          _isProcessingRecording = false;
          _yoloJsonPath = null;
          _framesAnalyzed = 0;
          _detections = 0;
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
      if (analysisDialogVisible && mounted) {
        Navigator.of(context, rootNavigator: true).pop();
      }
      if (!starting && mounted) {
        setState(() => _isProcessingRecording = false);
      }
    }
  }

  Future<_YoloSummary> _runNativeYoloAnalysis({
    required String videoPath,
    required Directory sessionDir,
  }) async {
    final result = await _analysisChannel.invokeMapMethod<String, dynamic>(
      'runYoloOnVideo',
      {
        'videoPath': videoPath,
        'sessionDir': sessionDir.path,
        'sampledFps': _sampledFps,
      },
    );

    if (result == null) {
      throw StateError('YOLO returned no data');
    }

    final jsonPath = result['jsonPath'] as String?;
    final frames = (result['framesProcessed'] as num?)?.toInt();
    final detections = (result['detections'] as num?)?.toInt();

    if (jsonPath == null || frames == null || detections == null) {
      throw StateError('YOLO response missing fields: $result');
    }

    return _YoloSummary(
      jsonPath: jsonPath,
      framesProcessed: frames,
      detections: detections,
    );
  }

  void _runAnalysis() {
    final videoPath = _recordedVideoPath;
    if (videoPath == null) return;

    final metrics = [
      FeedbackMetric(label: 'Frames Processed', value: '$_framesAnalyzed'),
      FeedbackMetric(label: 'Detections', value: '$_detections'),
      if (_yoloJsonPath != null)
        FeedbackMetric(label: 'Detections File', value: _yoloJsonPath!.split('/').last),
    ];

    final session = FeedbackSession(
      exercise: widget.exercise,
      videoPath: videoPath,
      metrics: metrics,
      summary: _detections > 0
          ? 'Detected $_detections objects across $_framesAnalyzed frames.'
          : 'No detections were found in the sampled frames.',
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

class _AnalyzingDialog extends StatelessWidget {
  const _AnalyzingDialog();

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async => false,
      child: AlertDialog(
        content: Row(
          mainAxisSize: MainAxisSize.min,
          children: const [
            CircularProgressIndicator(),
            SizedBox(width: 16),
            Text('Analyzing…'),
          ],
        ),
      ),
    );
  }
}

class _YoloSummary {
  const _YoloSummary({
    required this.jsonPath,
    required this.framesProcessed,
    required this.detections,
  });

  final String jsonPath;
  final int framesProcessed;
  final int detections;
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
