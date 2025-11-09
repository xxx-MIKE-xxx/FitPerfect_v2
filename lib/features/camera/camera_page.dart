import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../models/exercise.dart';
import '../../models/feedback_session.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key, required this.exercise});

  final Exercise exercise;

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  bool _isRecording = false;
  bool _hasRecording = false;
  int _repetitionCount = 0;

  void _toggleRecording() {
    setState(() {
      if (_isRecording) {
        _isRecording = false;
        _hasRecording = true;
        _repetitionCount = 12; // Placeholder for detected repetitions.
      } else {
        _isRecording = true;
        _hasRecording = false;
        _repetitionCount = 0;
      }
    });
  }

  void _runAnalysis() {
    final session = FeedbackSession(
      exercise: widget.exercise,
      videoPath: '/videos/${widget.exercise.id}_${DateTime.now().millisecondsSinceEpoch}.mp4',
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
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    const Center(
                      child: Icon(Icons.videocam, size: 96),
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
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _toggleRecording,
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
