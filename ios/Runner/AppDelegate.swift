import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let analyzer = VideoYOLOAnalyzer()

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)

    if let controller = window?.rootViewController as? FlutterViewController {
      let channel = FlutterMethodChannel(
        name: "fitperfect/analysis",
        binaryMessenger: controller.binaryMessenger
      )

      channel.setMethodCallHandler { [weak self] call, result in
        guard call.method == "runYoloOnVideo" else {
          result(FlutterMethodNotImplemented)
          return
        }

        guard
          let arguments = call.arguments as? [String: Any],
          let videoPath = arguments["videoPath"] as? String,
          let sessionDir = arguments["sessionDir"] as? String
        else {
          result(FlutterError(code: "invalid_args", message: "Missing videoPath/sessionDir", details: nil))
          return
        }

        let sampledFps = (arguments["sampledFps"] as? Double) ?? 3.0

        self?.analyzer.analyze(
          videoAtPath: videoPath,
          sessionDirectory: sessionDir,
          sampledFps: sampledFps
        ) { analysisResult in
          switch analysisResult {
          case let .success(summary):
            NSLog(
              "YOLO analysis complete. Frames: %d, detections: %d, json: %@",
              summary.totals.framesProcessed,
              summary.totals.detections,
              summary.jsonURL.path
            )
            DispatchQueue.main.async {
              result([
                "jsonPath": summary.jsonURL.path,
                "framesProcessed": summary.totals.framesProcessed,
                "detections": summary.totals.detections
              ])
            }
          case let .failure(error):
            NSLog("YOLO analysis failed: %@", error.localizedDescription)
            DispatchQueue.main.async {
              result(FlutterError(code: "yolo_failed", message: error.localizedDescription, details: nil))
            }
          }
        }
      }
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
