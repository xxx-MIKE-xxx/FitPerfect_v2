import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'features/camera/camera_page.dart';
import 'features/exercise_selection/exercise_selection_page.dart';
import 'features/feedback/feedback_page.dart';
import 'models/exercise.dart';
import 'models/feedback_session.dart';

GoRouter createRouter() => GoRouter(
      routes: [
        GoRoute(
          path: '/',
          builder: (_, __) => const ExerciseSelectionPage(),
        ),
        GoRoute(
          path: '/camera',
          builder: (context, state) {
            final exercise = state.extra as Exercise?;
            if (exercise == null) {
              return const MissingRouteDataPage(
                title: 'Exercise Missing',
                message:
                    'Return to the exercise list and choose an exercise to continue.',
              );
            }
            return CameraPage(exercise: exercise);
          },
        ),
        GoRoute(
          path: '/feedback',
          builder: (context, state) {
            final session = state.extra as FeedbackSession?;
            if (session == null) {
              return const MissingRouteDataPage(
                title: 'Session Missing',
                message: 'Record a session before viewing feedback.',
              );
            }
            return FeedbackPage(session: session);
          },
        ),
      ],
    );

class MissingRouteDataPage extends StatelessWidget {
  const MissingRouteDataPage({
    super.key,
    required this.title,
    required this.message,
  });

  final String title;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.warning_amber_rounded,
                  size: 64, color: Theme.of(context).colorScheme.error),
              const SizedBox(height: 16),
              Text(
                message,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: () => context.go('/'),
                icon: const Icon(Icons.home_outlined),
                label: const Text('Back to Exercises'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
