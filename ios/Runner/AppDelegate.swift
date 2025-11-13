import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let pipelineConfig: PipelineConfig = {
    do {
      return try PipelineConfig.load()
    } catch {
      fatalError("Failed to load pipeline configuration: \(error)")
    }
  }()

  private lazy var pipeline: VideoAnalysisPipeline = {
    VideoAnalysisPipeline(config: pipelineConfig)
  }()

  private var shouldLogBridgeArgumentKeys = true
  private var shouldLogBridgeResultKeys = true

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
        guard let self = self else {
          result(FlutterError(code: "app_delegate_missing", message: "AppDelegate unavailable", details: nil))
          return
        }

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

        logKeysOnce(arguments, tag: "BRIDGE_ARGS", onlyOnce: &self.shouldLogBridgeArgumentKeys)

        let sampledFps = (arguments["sampledFps"] as? Double) ?? 3.0
        let personStrategyString = (arguments["person"] as? String) ?? PersonSelectionStrategy.bestScore.rawValue
        let personStrategy = PersonSelectionStrategy(rawValue: personStrategyString) ?? .bestScore

        let rawPath = videoPath
        let videoURL: URL = rawPath.hasPrefix("file://") ? URL(string: rawPath)! : URL(fileURLWithPath: rawPath)
        let normalizedPath = videoURL.isFileURL ? videoURL.path : rawPath
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: normalizedPath, isDirectory: &isDir)
        NSLog("[BRIDGE] analyze request: raw=%@ normalized=%@ exists=%d dir=%d cfg_verbose=%d",
              rawPath,
              normalizedPath,
              exists ? 1 : 0,
              isDir.boolValue ? 1 : 0,
              self.pipeline.debugConfig.verboseLogging ? 1 : 0)

        guard exists, !isDir.boolValue else {
          result(["ok": false, "stage": "bridge", "error": "Video file not found at \(normalizedPath)"])
          return
        }

        NSLog("[BRIDGE] session=%@ sampledFps=%.3f", sessionDir, sampledFps)

        self.pipeline.run(
          videoPath: normalizedPath,
          sessionDirectory: sessionDir,
          sampledFps: sampledFps,
          personStrategy: personStrategy,
          progress: { status in
            DispatchQueue.main.async {
              channel.invokeMethod("analysisProgress", arguments: ["status": status])
            }
          },
          completion: { [weak self] pipelineResult in
            guard let self = self else { return }
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

                let post = summary.postprocess
                var postInfo: [String: Any] = [
                  "frames": post.totals.framesProcessed,
                  "framesWithDetections": post.totals.framesWithDetections,
                  "refinedPath": post.refinedURL.path,
                  "normalizedPath": post.normalizedURL.path,
                  "numKeypoints": post.numKeypoints
                ]
                postInfo["fps"] = post.fps
                postInfo["sampledFps"] = post.sampledFps

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
                  "rtmpose": poseInfo,
                  "postprocess": postInfo
                ]
                if let motionInfo = motionInfo {
                  response["motionbert"] = motionInfo
                }
                logKeysOnce(response, tag: "BRIDGE_RESULT", onlyOnce: &self.shouldLogBridgeResultKeys)
                result(response)
              case let .failure(error):
                let response: [String: Any] = [
                  "ok": false,
                  "stage": error.stage.rawValue,
                  "error": error.errorDescription ?? "Unknown error"
                ]
                logKeysOnce(response, tag: "BRIDGE_RESULT", onlyOnce: &self.shouldLogBridgeResultKeys)
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
