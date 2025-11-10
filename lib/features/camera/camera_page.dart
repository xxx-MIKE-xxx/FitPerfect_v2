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
  String? _poseJsonPath;
  String? _posePreviewPath;
  String? _motionJsonPath;
  String? _motionPreviewPath;
  int _framesAnalyzed = 0;
  int _detections = 0;
  int _poseFramesProcessed = 0;
  int _poseFramesWithDetections = 0;
  int _poseKeypoints = 0;
  int _motionFramesProcessed = 0;
  int _motionFramesWith3D = 0;
  String? _cameraError;
  final ValueNotifier<String> _analysisStatus = ValueNotifier<String>('Analyzing…');

  @override
  void initState() {
    super.initState();
    _analysisChannel.setMethodCallHandler(_handleAnalysisCallbacks);
    _initializeCamera();
  }

  @override
  void dispose() {
    _analysisChannel.setMethodCallHandler(null);
    _analysisStatus.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _handleAnalysisCallbacks(MethodCall call) async {
    if (call.method != 'analysisProgress') {
      return;
    }

    final args = call.arguments;
    String? status;
    if (args is Map) {
      final raw = args['status'];
      if (raw is String) {
        status = raw;
      }
    } else if (args is String) {
      status = args;
    }

    if (status != null) {
      _analysisStatus.value = status;
    }
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
          _analysisStatus.value = 'Analyzing…';
          // ignore: discarded_futures
          showDialog<void>(
            context: context,
            barrierDismissible: false,
            builder: (_) => _AnalyzingDialog(statusListenable: _analysisStatus),
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

        _AnalysisSummary? summary;
        try {
          summary = await _runNativeAnalysis(
            videoPath: savedMoviePath,
            sessionDir: activeSessionDir,
          );
        } on PlatformException catch (e) {
          final stage = e.details is Map ? (e.details['stage'] as String?) : null;
          final stageInfo = stage == null ? '' : ' (${stage.toUpperCase()})';
          _showSnackBar('Analysis$stageInfo failed: ${e.message ?? e.code}');
        } catch (e) {
          _showSnackBar('Analysis failed: $e');
        }

        if (!mounted) return;
        setState(() {
          _sessionDir = activeSessionDir;
          _isRecording = false;
          _hasRecording = true;
          _recordedVideoPath = savedMoviePath;
          _yoloJsonPath = summary?.yoloJsonPath;
          _poseJsonPath = summary?.poseJsonPath;
          _posePreviewPath = summary?.posePreviewPath;
          _motionJsonPath = summary?.motionJsonPath;
          _motionPreviewPath = summary?.motionPreviewPath;
          _framesAnalyzed = summary?.yoloFrames ?? 0;
          _detections = summary?.yoloDetections ?? 0;
          _poseFramesProcessed = summary?.poseFrames ?? 0;
          _poseFramesWithDetections = summary?.poseDetections ?? 0;
          _poseKeypoints = summary?.poseKeypoints ?? 0;
          _motionFramesProcessed = summary?.motionFrames ?? 0;
          _motionFramesWith3D = summary?.motionFramesWith3D ?? 0;
          _repetitionCount = summary?.yoloDetections ?? 0;
        });

        if (summary != null) {
          final motionFrames = summary.motionFrames ?? summary.poseFrames;
          final motionPart = summary.motionFramesWith3D == null
              ? ''
              : ' MotionBERT: ${summary.motionFramesWith3D}/${motionFrames ?? 0} frames with 3D.';
          final buffer = StringBuffer(
            'YOLO: ${summary.yoloDetections} detections across ${summary.yoloFrames} frames. '
            'RTMPose: ${summary.poseDetections}/${summary.poseFrames} frames with keypoints.',
          );
          buffer.write(motionPart);
          final message = buffer.toString();
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
          _poseJsonPath = null;
          _posePreviewPath = null;
          _motionJsonPath = null;
          _motionPreviewPath = null;
          _framesAnalyzed = 0;
          _detections = 0;
          _poseFramesProcessed = 0;
          _poseFramesWithDetections = 0;
          _poseKeypoints = 0;
          _motionFramesProcessed = 0;
          _motionFramesWith3D = 0;
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
        analysisDialogVisible = false;
      }
      if (!starting && mounted) {
        setState(() => _isProcessingRecording = false);
      }
      _analysisStatus.value = 'Analyzing…';
    }
  }

  Future<_AnalysisSummary> _runNativeAnalysis({
    required String videoPath,
    required Directory sessionDir,
  }) async {
    final result = await _analysisChannel.invokeMapMethod<String, dynamic>(
      'runVideoAnalysis',
      {
        'videoPath': videoPath,
        'sessionDir': sessionDir.path,
        'sampledFps': _sampledFps,
      },
    );

    if (result == null) {
      throw StateError('Analysis returned no data');
    }

    final ok = result['ok'] == true;
    if (!ok) {
      final stage = result['stage'] as String?;
      final errorMessage = result['error'] as String? ?? 'Unknown error';
      throw PlatformException(
        code: 'analysis_failed',
        message: errorMessage,
        details: {'stage': stage},
      );
    }

    final yolo = result['yolo'];
    final pose = result['rtmpose'];
    final motion = result['motionbert'];
    if (yolo is! Map || pose is! Map) {
      throw StateError('Analysis response missing stage summaries: $result');
    }

    final yoloJsonPath = yolo['jsonPath'] as String?;
    final yoloFrames = (yolo['frames'] as num?)?.toInt();
    final yoloDetections = (yolo['detections'] as num?)?.toInt();

    final poseJsonPath = pose['jsonPath'] as String?;
    final poseFrames = (pose['frames'] as num?)?.toInt();
    final poseDetections = (pose['framesWithDetections'] as num?)?.toInt();
    final poseKeypoints = (pose['numKeypoints'] as num?)?.toInt() ?? 0;
    final posePreviewPath = pose['previewPath'] as String?;

    String? motionJsonPath;
    int? motionFrames;
    int? motionFramesWith3D;
    String? motionPreviewPath;
    if (motion is Map) {
      motionJsonPath = motion['jsonPath'] as String?;
      motionFrames = (motion['frames'] as num?)?.toInt();
      motionFramesWith3D = (motion['framesWith3D'] as num?)?.toInt();
      motionPreviewPath = motion['previewPath'] as String?;
    }

    if (yoloJsonPath == null || yoloFrames == null || yoloDetections == null) {
      throw StateError('YOLO summary missing fields: $result');
    }
    if (poseJsonPath == null || poseFrames == null || poseDetections == null) {
      throw StateError('RTMPose summary missing fields: $result');
    }

    return _AnalysisSummary(
      yoloJsonPath: yoloJsonPath,
      yoloFrames: yoloFrames,
      yoloDetections: yoloDetections,
      poseJsonPath: poseJsonPath,
      poseFrames: poseFrames,
      poseDetections: poseDetections,
      poseKeypoints: poseKeypoints,
      posePreviewPath: posePreviewPath,
      motionJsonPath: motionJsonPath,
      motionFrames: motionFrames,
      motionFramesWith3D: motionFramesWith3D,
      motionPreviewPath: motionPreviewPath,
    );
  }

  void _runAnalysis() {
    final videoPath = _recordedVideoPath;
    if (videoPath == null) return;

    final metrics = [
      FeedbackMetric(label: 'YOLO Frames', value: '$_framesAnalyzed'),
      FeedbackMetric(label: 'YOLO Detections', value: '$_detections'),
      if (_yoloJsonPath != null)
        FeedbackMetric(label: 'Detections File', value: _yoloJsonPath!.split('/').last),
      FeedbackMetric(
        label: 'RTMPose Frames',
        value: '$_poseFramesProcessed',
      ),
      FeedbackMetric(
        label: 'Frames With Keypoints',
        value: '$_poseFramesWithDetections',
      ),
      if (_poseKeypoints > 0)
        FeedbackMetric(label: 'Keypoints Per Person', value: '$_poseKeypoints'),
      if (_poseJsonPath != null)
        FeedbackMetric(label: 'Pose File', value: _poseJsonPath!.split('/').last),
      if (_posePreviewPath != null)
        FeedbackMetric(label: 'Pose Preview', value: _posePreviewPath!.split('/').last),
      if (_motionFramesProcessed > 0)
        FeedbackMetric(label: 'MotionBERT Frames', value: '$_motionFramesProcessed'),
      if (_motionFramesWith3D > 0)
        FeedbackMetric(label: 'Frames With 3D', value: '$_motionFramesWith3D'),
      if (_motionJsonPath != null)
        FeedbackMetric(
          label: 'MotionBERT File',
          value: _motionJsonPath!.split('/').last,
        ),
      if (_motionPreviewPath != null)
        FeedbackMetric(
          label: 'MotionBERT Preview',
          value: _motionPreviewPath!.split('/').last,
        ),
    ];

    final sessionDirPath = _sessionDir?.path ?? File(videoPath).parent.path;

    final motionSummary = _motionFramesProcessed > 0
        ? ' MotionBERT lifted $_motionFramesWith3D of $_motionFramesProcessed frames into 3D.'
        : '';

    final session = FeedbackSession(
      exercise: widget.exercise,
      videoPath: videoPath,
      sessionDir: sessionDirPath,
      metrics: metrics,
      summary: (_poseFramesWithDetections > 0
              ? 'YOLO detected $_detections objects across $_framesAnalyzed frames. '
                  'RTMPose produced keypoints for $_poseFramesWithDetections of $_poseFramesProcessed frames.'
              : 'No pose keypoints were detected in the sampled frames.') +
          motionSummary,
      poseJsonPath: _poseJsonPath,
      yoloJsonPath: _yoloJsonPath,
      posePreviewPath: _posePreviewPath,
      motionJsonPath: _motionJsonPath,
      motionPreviewPath: _motionPreviewPath,
      motionFrames: _motionFramesProcessed,
      motionFramesWith3D: _motionFramesWith3D,
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
  const _AnalyzingDialog({required this.statusListenable});

  final ValueListenable<String> statusListenable;

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async => false,
      child: AlertDialog(
        content: ValueListenableBuilder<String>(
          valueListenable: statusListenable,
          builder: (context, status, _) {
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const CircularProgressIndicator(),
                const SizedBox(width: 16),
                Flexible(
                  child: Text(
                    status,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _AnalysisSummary {
  const _AnalysisSummary({
    required this.yoloJsonPath,
    required this.yoloFrames,
    required this.yoloDetections,
    required this.poseJsonPath,
    required this.poseFrames,
    required this.poseDetections,
    required this.poseKeypoints,
    this.posePreviewPath,
    this.motionJsonPath,
    this.motionFrames,
    this.motionFramesWith3D,
    this.motionPreviewPath,
  });

  final String yoloJsonPath;
  final int yoloFrames;
  final int yoloDetections;
  final String poseJsonPath;
  final int poseFrames;
  final int poseDetections;
  final int poseKeypoints;
  final String? posePreviewPath;
  final String? motionJsonPath;
  final int? motionFrames;
  final int? motionFramesWith3D;
  final String? motionPreviewPath;
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
