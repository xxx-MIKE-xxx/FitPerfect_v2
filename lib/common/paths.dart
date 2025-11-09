import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class Paths {
  static const _sessionsDirectoryName = 'sessions';

  static Future<Directory> makeNewSessionDir() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final sessionsRoot = Directory(
      p.join(documentsDir.path, _sessionsDirectoryName),
    );

    if (!await sessionsRoot.exists()) {
      await sessionsRoot.create(recursive: true);
    }

    Directory candidate;
    var attempt = 0;
    do {
      final timestamp = _timestampSuffix();
      final suffix = attempt == 0 ? '' : '-$attempt';
      candidate = Directory(
        p.join(sessionsRoot.path, 'session_$timestamp$suffix'),
      );
      attempt++;
    } while (await candidate.exists());

    await candidate.create(recursive: true);
    return candidate;
  }

  static File movieFile(Directory sessionDir) {
    return File(p.join(sessionDir.path, 'movie.mov'));
  }

  static String moviePath(Directory sessionDir) => movieFile(sessionDir).path;

  static Uri movieUri(Directory sessionDir) => Uri.file(moviePath(sessionDir));

  static String _timestampSuffix() {
    final now = DateTime.now().toUtc();
    final year = now.year.toString().padLeft(4, '0');
    final month = now.month.toString().padLeft(2, '0');
    final day = now.day.toString().padLeft(2, '0');
    final hour = now.hour.toString().padLeft(2, '0');
    final minute = now.minute.toString().padLeft(2, '0');
    final second = now.second.toString().padLeft(2, '0');
    return '${year}${month}${day}_${hour}${minute}${second}';
  }
}
