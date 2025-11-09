import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class Paths {
  static const String _sessionsFolderName = 'sessions';

  /// Creates a new directory for storing artifacts of a recording session.
  static Future<Directory> makeNewSessionDir() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
    final sessionDir = Directory(
      p.join(documentsDir.path, _sessionsFolderName, timestamp),
    );

    if (!await sessionDir.exists()) {
      await sessionDir.create(recursive: true);
    }

    return sessionDir;
  }

  /// Returns the URI where the raw movie file should be stored.
  static Uri movieUrl(Directory sessionDir) {
    final filePath = p.join(sessionDir.path, 'movie.mov');
    return Uri.file(filePath);
  }

  /// Convenience helper returning the file that backs [movieUrl].
  static File movieFile(Directory sessionDir) {
    return File(movieUrl(sessionDir).toFilePath());
  }
}
