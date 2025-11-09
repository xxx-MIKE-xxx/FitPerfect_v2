import 'dart:io';

import 'package:camera/camera.dart';
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
  Future<void>? _initializeCameraFuture;
  Directory? _currentSessionDir;
  Uri? _lastVideoUri;
  bool _isRecording = false;
  bool _hasRecording = false;
  bool _isProcessing = false;
  int _repetitionCount = 0;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (!mounted) return;
        setState(() {
          _errorMessage = 'No cameras available on this device.';
        });
        return;
      }

      final preferredCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        preferredCamera,
        ResolutionPreset.high,
        enableAudio: true,
      );

      final initializeFuture = controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _cameraController = controller;
        _initializeCameraFuture = initializeFuture;
        _errorMessage = null;
      });

      await initializeFuture;
      if (!mounted) {
        await controller.dispose();
      }
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to initialize the camera: $error';
      });
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    final controller = _cameraController;
    final initializeFuture = _initializeCameraFuture;
    if (controller == null || initializeFuture == null) {
      return;
    }

    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      await initializeFuture;

      if (_isRecording) {
        final recording = await controller.stopVideoRecording();
        final sessionDir = _currentSessionDir ?? await Paths.makeNewSessionDir();
        final movieFile = Paths.movieFile(sessionDir);
        await movieFile.parent.create(recursive: true);
        await recording.saveTo(movieFile.path);

        if (!mounted) return;
        setState(() {
          _isRecording = false;
          _hasRecording = true;
          _repetitionCount = 12; // Placeholder for detected repetitions.
          _lastVideoUri = movieFile.uri;
          _currentSessionDir = sessionDir;
        });
      } else {
        final sessionDir = await Paths.makeNewSessionDir();
        await controller.startVideoRecording();

        if (!mounted) return;
        setState(() {
          _isRecording = true;
          _hasRecording = false;
          _repetitionCount = 0;
          _currentSessionDir = sessionDir;
          _lastVideoUri = null;
        });
      }
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Unable to ${_isRecording ? 'stop' : 'start'} recording: $error')),
      );
      setState(() {
        _isRecording = false;
        _hasRecording = false;
      });
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  void _runAnalysis() {
    final videoUri = _lastVideoUri;
    if (videoUri == null) return;

    final session = FeedbackSession(
      exercise: widget.exercise,
      videoPath: videoUri.toFilePath(),
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
              child: DecoratedBox(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(24),
                  color: Theme.of(context).colorScheme.surfaceVariant,
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (_errorMessage != null)
                        Center(
                          child: Text(
                            _errorMessage!,
                            style: Theme.of(context)
                                .textTheme
                                .bodyLarge
                                ?.copyWith(color: Colors.white),
                            textAlign: TextAlign.center,
                          ),
                        )
                      else if (_initializeCameraFuture == null)
                        const Center(child: CircularProgressIndicator())
                      else
                        FutureBuilder<void>(
                          future: _initializeCameraFuture,
                          builder: (context, snapshot) {
                            if (snapshot.connectionState == ConnectionState.done &&
                                _cameraController != null) {
                              return CameraPreview(_cameraController!);
                            }
                            if (snapshot.hasError) {
                              return Center(
                                child: Text(
                                  'Camera error: ${snapshot.error}',
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodyLarge
                                      ?.copyWith(color: Colors.white),
                                  textAlign: TextAlign.center,
                                ),
                              );
                            }
                            return const Center(child: CircularProgressIndicator());
                          },
                        ),
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
                                  Text(
                                    _isRecording ? 'Recording…' : 'Idle',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                    ),
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
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: (_initializeCameraFuture == null || _isProcessing)
                  ? null
                  : () => _toggleRecording(),
              icon: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record),
              label: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
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
