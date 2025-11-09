import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

GoRouter createRouter() => GoRouter(routes: [
  GoRoute(path: '/', builder: (_, __) => const HomePage(), routes: [
    GoRoute(path: 'settings', builder: (_, __) => const SettingsPage()),
  ]),
]);

class HomePage extends StatelessWidget {
  const HomePage({super.key});
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Home')),
    body: Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: const [
        Text('Flutter baseline ready ✅'),
      ]),
    ),
  );
}

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Settings')),
    body: const Center(child: Text('Settings placeholder')),
  );
}
