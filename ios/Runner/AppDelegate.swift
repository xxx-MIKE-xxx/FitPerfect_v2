import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let pipeline = VideoAnalysisPipeline()

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
        guard call.method == "runVideoAnalysis" else {
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
        let personStrategyString = (arguments["person"] as? String) ?? PersonSelectionStrategy.bestScore.rawValue
        let personStrategy = PersonSelectionStrategy(rawValue: personStrategyString) ?? .bestScore

        self?.pipeline.run(
          videoPath: videoPath,
          sessionDirectory: sessionDir,
          sampledFps: sampledFps,
          personStrategy: personStrategy,
          progress: { status in
            DispatchQueue.main.async {
              channel.invokeMethod("analysisProgress", arguments: ["status": status])
            }
          },
          completion: { pipelineResult in
            DispatchQueue.main.async {
              switch pipelineResult {
              case let .success(summary):
                var poseInfo: [String: Any] = [
                  "frames": summary.rtmpose.totals.framesProcessed,
                  "framesWithDetections": summary.rtmpose.totals.framesWithDetections,
                  "jsonPath": summary.rtmpose.jsonURL.path,
                  "numKeypoints": summary.rtmpose.numKeypoints
                ]
                if let preview = summary.rtmpose.previewURL {
                  poseInfo["previewPath"] = preview.path
                }

                var motionInfo: [String: Any]?
                if let motion = summary.motionbert {
                  var info: [String: Any] = [
                    "frames": motion.totals.framesProcessed,
                    "framesWith3D": motion.totals.framesWith3D,
                    "jsonPath": motion.jsonURL.path
                  ]
                  if let preview = motion.previewURL {
                    info["previewPath"] = preview.path
                  }
                  motionInfo = info
                }

                var response: [String: Any] = [
                  "ok": true,
                  "yolo": [
                    "frames": summary.yolo.totals.framesProcessed,
                    "detections": summary.yolo.totals.detections,
                    "jsonPath": summary.yolo.jsonURL.path
                  ],
                  "rtmpose": poseInfo
                ]
                if let motionInfo = motionInfo {
                  response["motionbert"] = motionInfo
                }
                result(response)
              case let .failure(error):
                let response: [String: Any] = [
                  "ok": false,
                  "stage": error.stage.rawValue,
                  "error": error.errorDescription ?? "Unknown error"
                ]
                result(response)
              }
            }
          }
        )
      }
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
