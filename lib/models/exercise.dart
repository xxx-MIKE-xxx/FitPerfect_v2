import 'package:flutter/foundation.dart';

@immutable
class Exercise {
  final String id;
  final String name;
  final String description;
  final String focusArea;

  const Exercise({
    required this.id,
    required this.name,
    required this.description,
    required this.focusArea,
  });
}

const sampleExercises = <Exercise>[
  Exercise(
    id: 'squat',
    name: 'Bodyweight Squat',
    description: 'Track depth, knee alignment, and hip drive.',
    focusArea: 'Lower Body',
  ),
  Exercise(
    id: 'pushup',
    name: 'Push-Up',
    description: 'Monitor posture, elbow flare, and tempo.',
    focusArea: 'Upper Body',
  ),
  Exercise(
    id: 'lunge',
    name: 'Reverse Lunge',
    description: 'Assess balance, stride length, and torso stability.',
    focusArea: 'Legs & Core',
  ),
  Exercise(
    id: 'plank',
    name: 'Plank Reach',
    description: 'Measure hip sway and shoulder stability.',
    focusArea: 'Core',
  ),
];
