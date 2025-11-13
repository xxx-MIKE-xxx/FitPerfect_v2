import Foundation
import AVFoundation
import CoreML
import Vision
import UIKit

// =====================================================
// MARK: – Pipeline configuration (no calibration here)
// =====================================================

private let YOLO_BBOX_MODE               = "xywh_tl"
private let YOLO_COORDS_ORIGIN           = "top_left"

// =====================================================
// MARK: – YOLO stage types
// =====================================================

enum PersonSelectionStrategy: String {
    case bestScore
    case largest
}

struct YOLODetectionBox: Codable {
    let cls: Int
    let label: String
    let score: Double
    let x: Int
    let y: Int
    let w: Int
    let h: Int
}

struct YOLOFrameSelection: Codable {
    let strategy: String
    let index: Int
}

struct YOLODetectionFrame: Codable {
    let fi: Int
    let t: Double
    let boxes: [YOLODetectionBox]
    let selected: YOLOFrameSelection?
}

struct YOLODetectionMeta: Codable {
    let bboxMode: String
    let coordsOrigin: String
    let rotationCorrectionDeg: Int
    let channelOrder: String

    enum CodingKeys: String, CodingKey {
        case bboxMode = "bbox_mode"
        case coordsOrigin = "coords_origin"
        case rotationCorrectionDeg = "rotation_correction_deg"
        case channelOrder = "channel_order"
    }
}

struct YOLODetectionTotals: Codable {
    let framesProcessed: Int
    let detections: Int
}

struct YOLODetectionOutput: Codable {
    let videoWidth: Int
    let videoHeight: Int
    let fps: Double
    let sampledFps: Double
    let meta: YOLODetectionMeta?
    let frames: [YOLODetectionFrame]
    let totals: YOLODetectionTotals
}

struct YOLOAnalysisSummary {
    let jsonURL: URL
    let totals: YOLODetectionTotals
}

enum YOLOAnalysisError: Error {
    case modelUnavailable
    case videoTrackUnavailable
    case failedToWriteJSON
}

// =====================================================
// MARK: – YOLO stage
// =====================================================

final class VideoYOLOAnalyzer {
    private let config: PipelineConfig
    private let yoloConfig: PipelineConfig.YOLO
    private let debugConfig: PipelineConfig.Debug
    private var asset: AVAsset?
    private var track: AVAssetTrack?
    private var model: MLModel?
    private var visionModel: VNCoreMLModel?
    private var generator: AVAssetImageGenerator?
    private let personStrategy: PersonSelectionStrategy
    private let frameRotation: Int

    private let classLabels: [String] = [
        "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
        "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
        "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"
    ]

    init(config: PipelineConfig, personStrategy: PersonSelectionStrategy = .bestScore) {
        self.config = config
        self.yoloConfig = config.yolo
        self.debugConfig = config.debug
        self.personStrategy = personStrategy
        self.frameRotation = normalizedRotation(config.yolo.rotationCorrectionDeg)
    }

    func analyze(
        videoAtPath videoPath: String,
        sessionDirectory: String,
        sampledFps: Double
    ) throws -> YOLOAnalysisSummary {
        let videoURL = URL(fileURLWithPath: videoPath)
        let asset = AVAsset(url: videoURL)
        self.asset = asset

        let preloadKeys = ["tracks", "duration", "playable"]
        let preloadGroup = DispatchGroup()
        preloadGroup.enter()
        asset.loadValuesAsynchronously(forKeys: preloadKeys) {
            preloadGroup.leave()
        }
        preloadGroup.wait()
        for key in preloadKeys {
            var error: NSError?
            let status = asset.statusOfValue(forKey: key, error: &error)
            NSLog("[YOLO] asset key '%@' status=%d err=%@", key, status.rawValue, String(describing: error))
        }

        guard let track = asset.tracks(withMediaType: .video).first else {
            throw YOLOAnalysisError.videoTrackUnavailable
        }
        self.track = track

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let mlmodelcURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc")
        let mlpackageURL = Bundle.main.url(forResource: "best", withExtension: "mlpackage")
        guard let modelURL = mlpackageURL ?? mlmodelcURL else {
            NSLog("[YOLO] ERROR model resource missing mlmodelc=%@ mlpackage=%@",
                  mlmodelcURL?.path ?? "nil", mlpackageURL?.path ?? "nil")
            throw YOLOAnalysisError.modelUnavailable
        }
        let yoloModel = try MLModel.load(contentsOf: modelURL, configuration: configuration)
        model = yoloModel
        visionModel = try VNCoreMLModel(for: yoloModel)
        let modelSource = modelURL.pathExtension.lowercased() == "mlmodelc" ? "mlmodelc" : "mlpackage"
        var inputSummary = "unknown"
        if let desc = yoloModel.modelDescription.inputDescriptionsByName.values.first {
            if let constraint = desc.imageConstraint {
                inputSummary = "\(constraint.pixelsWide)x\(constraint.pixelsHigh)"
            } else if let multi = desc.multiArrayConstraint {
                let shape = multi.shape.map { Int(truncating: $0) }
                inputSummary = shape.map { String($0) }.joined(separator: "x")
            }
        }
        NSLog("[YOLO] model loaded (%@) computeUnits=%@ input=%@", modelSource, "all", inputSummary)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceAfter = .zero
        generator.requestedTimeToleranceBefore = .zero
        self.generator = generator

        let transformedSize = track.naturalSize.applying(track.preferredTransform)
        let videoWidth = Int(abs(transformedSize.width))
        let videoHeight = Int(abs(transformedSize.height))

        let fps = track.nominalFrameRate > 0 ? Double(track.nominalFrameRate) : estimateFrameRate(for: track)
        let durationSeconds = CMTimeGetSeconds(asset.duration)

        let targetSampledFps = max(sampledFps, 1.0)
        let baseFps = fps > 0 ? fps : 30.0
        let frameStep = max(1, Int(round(baseFps / targetSampledFps)))
        let effectiveSampledFps = baseFps / Double(frameStep)
        let frameDuration = 1.0 / baseFps

        var frames: [YOLODetectionFrame] = []
        var totalDetections = 0

        let timeScale = asset.duration.timescale != 0 ? asset.duration.timescale : 600

        let rotation = frameRotation

        var sampleIndex = 0
        NSLog("[YOLO] video=%dx%d duration=%.3fs baseFps=%.3f step=%d effective=%.3f",
              videoWidth, videoHeight, durationSeconds, baseFps, frameStep, effectiveSampledFps)
        if abs(effectiveSampledFps - targetSampledFps) > 0.001 {
            NSLog("[YOLO] requested sampledFps=%.3f adjusted to %.3f", targetSampledFps, effectiveSampledFps)
        }
        var failedFrames = 0
        while true {
            let fi = sampleIndex * frameStep
            let timestamp = Double(fi) * frameDuration
            if timestamp > durationSeconds + frameDuration * 0.5 {
                break
            }
            autoreleasepool {
                let time = CMTime(seconds: timestamp, preferredTimescale: timeScale)
                if let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) {
                    let uprightImage = self.rotateForYOLOIfNeeded(cgImage, rotation: rotation)
                    let frame = self.processFrame(
                        cgImage: uprightImage,
                        frameIndex: fi,
                        timestamp: timestamp,
                        videoWidth: videoWidth,
                        videoHeight: videoHeight,
                        rotation: rotation
                    )
                    totalDetections += frame.boxes.count
                    frames.append(frame)
                } else {
                    failedFrames += 1
                    NSLog("[YOLO] WARN fi=%d could not extract frame at t=%.3f", fi, timestamp)
                    // *** FIX: always include `selected` ***
                    let emptyFrame = YOLODetectionFrame(
                        fi: fi,
                        t: round(timestamp * 100) / 100,
                        boxes: [],
                        selected: YOLOFrameSelection(strategy: personStrategy.rawValue, index: -1)
                    )
                    frames.append(emptyFrame)
                }
            }
            sampleIndex += 1
        }

        if frames.isEmpty {
            let reason: String
            if durationSeconds <= 0 {
                reason = "invalid_duration"
            } else if baseFps <= 0 {
                reason = "invalid_fps"
            } else if failedFrames > 0 {
                reason = "frame_extraction_failed"
            } else {
                reason = "unknown"
            }
            NSLog("[YOLO] WARN no frames processed (reason=%@ duration=%.3f baseFps=%.3f step=%d)",
                  reason, durationSeconds, baseFps, frameStep)
        }

        let totals = YOLODetectionTotals(framesProcessed: frames.count, detections: totalDetections)

        // Store meta so RTMPose can mirror what we used in Python
        let meta = YOLODetectionMeta(
            bboxMode: yoloConfig.bboxMode,
            coordsOrigin: yoloConfig.coordsOrigin,
            rotationCorrectionDeg: frameRotation,
            channelOrder: config.rtmpose.channelOrder
        )

        let output = YOLODetectionOutput(
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            fps: fps,
            sampledFps: effectiveSampledFps,
            meta: meta,
            frames: frames,
            totals: totals
        )

        let sessionURL = URL(fileURLWithPath: sessionDirectory, isDirectory: true)
        let jsonURL = sessionURL.appendingPathComponent("yolo_detections.json")

        try FileManager.default.createDirectory(at: sessionURL, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(output)
            try data.write(to: jsonURL, options: .atomic)
        } catch {
            throw YOLOAnalysisError.failedToWriteJSON
        }

        // Debug banner (easy to spot in Xcode console)
        NSLog("[YOLO] video=%dx%d fps=%.2f sampledFps=%.2f   meta{ bbox=%@, origin=%@, rot=%d, chan=%@ }",
              videoWidth, videoHeight, fps, effectiveSampledFps,
              meta.bboxMode, meta.coordsOrigin, meta.rotationCorrectionDeg, meta.channelOrder)

        return YOLOAnalysisSummary(jsonURL: jsonURL, totals: totals)
    }

    func teardown() {
        generator = nil
        visionModel = nil
        model = nil
        track = nil
        asset = nil
    }

    private func rotateForYOLOIfNeeded(_ image: CGImage, rotation: Int) -> CGImage {
        guard rotation != 0 else { return image }
        return rotateImage(image, clockwiseDegrees: rotation) ?? image
    }

    private func processFrame(
        cgImage: CGImage,
        frameIndex: Int,
        timestamp: Double,
        videoWidth: Int,
        videoHeight: Int,
        rotation: Int
    ) -> YOLODetectionFrame {
        guard let visionModel = visionModel else {
            return YOLODetectionFrame(
                fi: frameIndex,
                t: round(timestamp * 100) / 100,
                boxes: [],
                selected: YOLOFrameSelection(strategy: personStrategy.rawValue, index: -1)
            )
        }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        do {
            try handler.perform([request])
            guard let observations = request.results as? [VNRecognizedObjectObservation] else {
                NSLog("[YOLO] fi=%d no recognized objects (no VN results)", frameIndex)
                // *** FIX: always include `selected` ***
                return YOLODetectionFrame(
                    fi: frameIndex,
                    t: round(timestamp * 100) / 100,
                    boxes: [],
                    selected: YOLOFrameSelection(strategy: personStrategy.rawValue, index: -1)
                )
            }
            if observations.isEmpty {
                NSLog("[YOLO] fi=%d no recognized objects", frameIndex)
            }
            let rotatedWidth = cgImage.width
            let rotatedHeight = cgImage.height
            var boxes: [YOLODetectionBox] = []
            for observation in observations {
                guard let topLabel = observation.labels.first else { continue }
                let label = topLabel.identifier
                let clsIndex = classLabels.firstIndex(of: label) ?? -1

                // VN boxes are normalized in Vision coords: (0,0) bottom-left
                let rect = observation.boundingBox
                let width = Double(rotatedWidth)
                let height = Double(rotatedHeight)

                let xRot = rect.minX * width
                let yRot = (1.0 - rect.maxY) * height
                let wRot = rect.width * width
                let hRot = rect.height * height

                let rotatedRect = CGRect(x: xRot, y: yRot, width: wRot, height: hRot)
                let mappedRect = mapRectFromRotated(
                    rotatedRect,
                    rotation: rotation,
                    originalWidth: videoWidth,
                    originalHeight: videoHeight
                )
                let clamped = clampRectToBounds(mappedRect, width: videoWidth, height: videoHeight)

                boxes.append(
                    YOLODetectionBox(
                        cls: clsIndex,
                        label: label,
                        score: Double(topLabel.confidence),
                        x: Int(clamped.origin.x.rounded()),
                        y: Int(clamped.origin.y.rounded()),
                        w: Int(clamped.width.rounded()),
                        h: Int(clamped.height.rounded())
                    )
                )
            }

            let selectedIndex = self.selectPersonIndex(in: boxes)
            let selection = YOLOFrameSelection(strategy: personStrategy.rawValue, index: selectedIndex)

            if debugConfig.verboseLogging && every(frameIndex, debugConfig.frameLogStride) {
                let preview = boxes.prefix(2).map {
                    "\($0.label)#\($0.cls) s=\(String(format: "%.2f", $0.score)) xywh=(\($0.x),\($0.y),\($0.w),\($0.h))"
                }.joined(separator: " | ")
                NSLog("[YOLO] fi=%d boxes=%d sel=%d %@", frameIndex, boxes.count, selectedIndex, preview)
            }

            return YOLODetectionFrame(
                fi: frameIndex,
                t: round(timestamp * 100) / 100,
                boxes: boxes,
                selected: selection
            )
        } catch {
            NSLog("[YOLO] fi=%d vision error=%@", frameIndex, error.localizedDescription)
            return YOLODetectionFrame(
                fi: frameIndex,
                t: round(timestamp * 100) / 100,
                boxes: [],
                selected: YOLOFrameSelection(strategy: personStrategy.rawValue, index: -1)
            )
        }
    }

    private func selectPersonIndex(in boxes: [YOLODetectionBox]) -> Int {
        var bestIndex: Int = -1
        var bestScore: Double = -Double.greatestFiniteMagnitude
        var bestArea: Int = Int.min

        for (index, box) in boxes.enumerated() where box.label == "person" {
            switch personStrategy {
            case .bestScore:
                if box.score > bestScore {
                    bestScore = box.score
                    bestIndex = index
                }
            case .largest:
                let area = box.w * box.h
                if area > bestArea {
                    bestArea = area
                    bestIndex = index
                }
            }
        }
        return bestIndex
    }

    private func estimateFrameRate(for track: AVAssetTrack) -> Double {
        if track.nominalFrameRate > 0 { return Double(track.nominalFrameRate) }
        let minFrameDuration = track.minFrameDuration
        if minFrameDuration.isValid && minFrameDuration.value != 0 {
            return Double(minFrameDuration.timescale) / Double(minFrameDuration.value)
        }
        return 0
    }
}

// =====================================================
// MARK: – RTMPose stage types
// =====================================================

enum RTMPoseError: Error {
    case modelUnavailable
    case failedToLoadYOLO
    case failedToWriteJSON
    case invalidOutputs
    case failedToCreatePreview
}

struct RTMPoseKeypoint: Codable {
    let x: Double
    let y: Double
    let score: Double
}

struct RTMPoseFrame: Codable {
    let fi: Int
    let t: Double
    let ok: Bool
    let keypoints: [RTMPoseKeypoint]
}

struct RTMPoseTotals: Codable {
    let framesProcessed: Int
    let framesWithDetections: Int
}

struct RTMPoseJSONOutput: Codable {
    struct InputSize: Codable {
        let w: Int
        let h: Int
    }

    struct Debug: Codable {
        let simccXShape: [Int]
        let simccYShape: [Int]
        enum CodingKeys: String, CodingKey {
            case simccXShape = "simcc_x_shape"
            case simccYShape = "simcc_y_shape"
        }
    }

    let videoWidth: Int
    let videoHeight: Int
    let fps: Double
    let sampledFps: Double
    let numKeypoints: Int
    let simccRatio: Double
    let inputSize: InputSize
    let frames: [RTMPoseFrame]
    let totals: RTMPoseTotals
    let debug: Debug?
}

struct RTMPoseSummary {
    let jsonURL: URL
    let previewURL: URL?
    let totals: RTMPoseTotals
    let numKeypoints: Int
}

// =====================================================
// MARK: – RTMPose (deterministic, no calibration)
// =====================================================

final class RTMPoseAnalyzer {
    private let config: PipelineConfig
    private let poseConfig: PipelineConfig.RTMPose
    private let paddingFactor: Double
    private let personStrategy: PersonSelectionStrategy
    private let rotationOverrideDegrees: Int
    private let inputWidth: Int
    private let inputHeight: Int
    private let simccRatio: Double
    private let means: [Float]
    private let stds: [Float]
    private let minKeypointScore: Double
    private let debugConfig: PipelineConfig.Debug

    private var model: RTMPose?
    private var asset: AVAsset?
    private var generator: AVAssetImageGenerator?
    private var lastSimccShapes: ([Int], [Int])?
    private var shouldLogSimccShapes: Bool
    private var loggedSimccShapeWarning = false

    // skeleton only for preview overlay
    private let skeletonPairs: [(Int, Int)] = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16), (5, 1), (6, 2), (1, 3), (2, 4)
    ]

    init(config: PipelineConfig, personStrategy: PersonSelectionStrategy = .bestScore) {
        self.config = config
        self.poseConfig = config.rtmpose
        self.paddingFactor = config.rtmpose.padding
        self.personStrategy = personStrategy
        self.rotationOverrideDegrees = config.rtmpose.rotationOverrideDeg
        self.inputWidth = config.rtmpose.inputWidth
        self.inputHeight = config.rtmpose.inputHeight
        self.simccRatio = config.rtmpose.simccRatio
        self.means = config.rtmpose.means.map { Float($0) }
        self.stds = config.rtmpose.stds.map { Float($0) }
        self.minKeypointScore = config.rtmpose.minKeypointScore
        self.debugConfig = config.debug
        self.shouldLogSimccShapes = config.debug.dumpModelOutputShapesOnce
    }

    func analyze(
        videoAtPath videoPath: String,
        sessionDirectory: String,
        yoloJSONURL: URL
    ) throws -> RTMPoseSummary {
        let jsonData = try Data(contentsOf: yoloJSONURL)
        let yolo = try JSONDecoder().decode(YOLODetectionOutput.self, from: jsonData)

        lastSimccShapes = nil
        shouldLogSimccShapes = debugConfig.dumpModelOutputShapesOnce
        loggedSimccShapeWarning = false

        // Video + model
        let videoURL = URL(fileURLWithPath: videoPath)
        let asset = AVAsset(url: videoURL)
        self.asset = asset

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        self.model = try RTMPose(configuration: configuration)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        self.generator = generator

        var frames: [RTMPoseFrame] = []
        var framesWithDetections = 0
        var previewURL: URL?
        var maxK = 0

        let sessionURL = URL(fileURLWithPath: sessionDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: sessionURL, withIntermediateDirectories: true)
        let poseJSONURL = sessionURL.appendingPathComponent("rtmpose_keypoints.json")
        let previewOutputURL = sessionURL.appendingPathComponent("rtmpose_preview.jpg")

        let timeScale = asset.duration.timescale != 0 ? asset.duration.timescale : 600

        // Deterministic rotation from YOLO meta + override constant
        let metaRotation = yolo.meta?.rotationCorrectionDeg ?? 0
        let effectiveRotation = normalizedRotation(metaRotation + rotationOverrideDegrees)

        if rotationOverrideDegrees != 0 {
            NSLog("[RTMPose] rotation override=%d", rotationOverrideDegrees)
        }
        NSLog("[RTMPose] input=%dx%d simccRatio=%.1f pad=%.2f rot(meta=%d + override=%d -> used=%d) channel=%@",
              inputWidth, inputHeight, simccRatio, paddingFactor,
              metaRotation, rotationOverrideDegrees, effectiveRotation, poseConfig.channelOrder)

        for fr in yolo.frames {
            autoreleasepool {
                let hint = fr.selected?.index ?? -1
                guard let box = self.pickPerson(fr.boxes, hint: hint) else {
                    frames.append(RTMPoseFrame(fi: fr.fi, t: fr.t, ok: false, keypoints: []))
                    return
                }

                let time = CMTime(seconds: fr.t, preferredTimescale: timeScale)
                guard let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) else {
                    frames.append(RTMPoseFrame(fi: fr.fi, t: fr.t, ok: false, keypoints: [])); return
                }

                guard
                    let prep = self.prepareInput(
                        from: cgImage,
                        box: box,
                        videoW: yolo.videoWidth,
                        videoH: yolo.videoHeight,
                        rotation: effectiveRotation
                    ),
                    let out = try? self.predict(prep.array)
                else {
                    frames.append(RTMPoseFrame(fi: fr.fi, t: fr.t, ok: false, keypoints: [])); return
                }

                let kps = self.decode(out, cropRect: prep.cropRect,
                                      videoW: yolo.videoWidth, videoH: yolo.videoHeight,
                                      rotation: prep.rotation)

                if kps.isEmpty {
                    frames.append(RTMPoseFrame(fi: fr.fi, t: fr.t, ok: false, keypoints: []))
                    return
                }

                maxK = max(maxK, kps.count)
                let isStrong = kps.contains { $0.score >= minKeypointScore }
                frames.append(RTMPoseFrame(fi: fr.fi, t: fr.t, ok: isStrong, keypoints: kps))
                if isStrong { framesWithDetections += 1 }

                if debugConfig.verboseLogging && every(fr.fi, debugConfig.frameLogStride) {
                    let strong = kps.filter { $0.score >= minKeypointScore }.count
                    NSLog("[RTMPose] fi=%d ok=%d strong=%d bbox=(%d,%d,%d,%d) cropPad=%.2f rot=%d",
                          fr.fi, isStrong ? 1 : 0, strong, box.x, box.y, box.w, box.h, paddingFactor, effectiveRotation)
                }

                if isStrong, previewURL == nil, let p = try? self.renderPreview(baseImage: cgImage, keypoints: kps, outputURL: previewOutputURL) {
                    previewURL = p
                    NSLog("[RTMPose] preview=%@", p.lastPathComponent)
                }
            }
        }

        let totals = RTMPoseTotals(framesProcessed: yolo.frames.count, framesWithDetections: framesWithDetections)

        let output = RTMPoseJSONOutput(
            videoWidth: yolo.videoWidth,
            videoHeight: yolo.videoHeight,
            fps: yolo.fps,
            sampledFps: yolo.sampledFps,
            numKeypoints: maxK,
            simccRatio: simccRatio,
            inputSize: .init(w: inputWidth, h: inputHeight),
            frames: frames,
            totals: totals,
            debug: lastSimccShapes.map { RTMPoseJSONOutput.Debug(simccXShape: $0.0, simccYShape: $0.1) }
        )

        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(output).write(to: poseJSONURL, options: .atomic)

        NSLog("[RTMPose] frames=%d ok=%d  (meta rot=%d, used rot=%d, chan=%@, pad=%.2f)",
              totals.framesProcessed, totals.framesWithDetections,
              yolo.meta?.rotationCorrectionDeg ?? 0, effectiveRotation,
              poseConfig.channelOrder, paddingFactor)

        return RTMPoseSummary(jsonURL: poseJSONURL, previewURL: previewURL, totals: totals, numKeypoints: maxK)
    }

    func teardown() {
        generator = nil
        asset = nil
        model = nil
        lastSimccShapes = nil
        shouldLogSimccShapes = false
        loggedSimccShapeWarning = false
    }

    // ---------------- helpers ----------------

    private func pickPerson(_ boxes: [YOLODetectionBox], hint: Int) -> YOLODetectionBox? {
        if hint >= 0, hint < boxes.count, boxes[hint].label == "person" { return boxes[hint] }
        let ppl = boxes.filter { $0.label == "person" }
        guard !ppl.isEmpty else { return nil }
        switch personStrategy {
        case .bestScore: return ppl.max(by: { $0.score < $1.score })
        case .largest:   return ppl.max(by: { ($0.w * $0.h) < ($1.w * $1.h) })
        }
    }

    private func prepareInput(
        from image: CGImage,
        box: YOLODetectionBox,
        videoW: Int,
        videoH: Int,
        rotation: Int
    ) -> (array: MLMultiArray, cropRect: CGRect, rotation: Int)? {

        let crop = paddedRect(for: box, videoW: videoW, videoH: videoH, pad: paddingFactor)

        let imageH = image.height
        let cgCrop = CGRect(
            x: crop.minX,
            y: Double(imageH) - crop.minY - crop.height,
            width: crop.width,
            height: crop.height
        ).integral

        guard let cropped = image.cropping(to: cgCrop) else { return nil }

        let rotated: CGImage
        if rotation != 0, let r = rotateImage(cropped, clockwiseDegrees: rotation) {
            rotated = r
        } else {
            rotated = cropped
        }

        // Draw into BGRA buffer at model input size
        var pixels = [UInt8](repeating: 0, count: inputWidth * inputHeight * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = inputWidth * 4
        let bmpInfo: CGBitmapInfo = [ .byteOrder32Little,
                                      CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue) ]
        guard let ctx = CGContext(
            data: &pixels,
            width: inputWidth, height: inputHeight,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: cs, bitmapInfo: bmpInfo.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(rotated, in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))

        // Build 1x3xH×W in the requested channel order (BGR or RGB)
        let N = inputWidth * inputHeight
        var floats = [Float](repeating: 0, count: N * 3)

        for i in 0..<N {
            let p = i * 4
            // BGRA memory layout
            let b = Float(pixels[p + 0])
            let g = Float(pixels[p + 1])
            let r = Float(pixels[p + 2])

            if poseConfig.channelOrder.uppercased() == "BGR" {
                // Channel 0 ← B, 1 ← G, 2 ← R  (means/stds applied in this order)
                floats[i]          = (b - means[0]) / stds[0]
                floats[N + i]      = (g - means[1]) / stds[1]
                floats[2 * N + i]  = (r - means[2]) / stds[2]
            } else {
                // Channel 0 ← R, 1 ← G, 2 ← B
                floats[i]          = (r - means[0]) / stds[0]
                floats[N + i]      = (g - means[1]) / stds[1]
                floats[2 * N + i]  = (b - means[2]) / stds[2]
            }
        }

        let shape: [NSNumber] = [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)]
        guard let arr = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }
        floats.withUnsafeBytes { buf in
            memcpy(arr.dataPointer, buf.baseAddress!, floats.count * MemoryLayout<Float>.size)
        }
        return (arr, crop, rotation)
    }

    private func predict(_ input: MLMultiArray) throws -> RTMPoseOutput {
        guard let model = model else { throw RTMPoseError.modelUnavailable }
        let inp = RTMPoseInput(input: input)
        return try model.prediction(input: inp)
    }

    private func decode(
        _ out: RTMPoseOutput,
        cropRect: CGRect,
        videoW: Int,
        videoH: Int,
        rotation: Int
    ) -> [RTMPoseKeypoint] {

        let X = out.simcc_x, Y = out.simcc_y
        let dx = X.shape.map { Int(truncating: $0) }
        let dy = Y.shape.map { Int(truncating: $0) }
        let sx = X.strides.map { Int(truncating: $0) }
        let sy = Y.strides.map { Int(truncating: $0) }
        lastSimccShapes = (dx, dy)
        if shouldLogSimccShapes {
            shouldLogSimccShapes = false
            NSLog("[RTMPose] simcc_x=%@ simcc_y=%@", String(describing: dx), String(describing: dy))
            let allowed: Set<Int> = [17, 26, 29]
            if !loggedSimccShapeWarning {
                let combined = dx + dy
                if combined.first(where: { allowed.contains($0) }) == nil {
                    NSLog("[RTMPose] WARN unexpected simcc keypoint shape=%@", String(describing: combined))
                    loggedSimccShapeWarning = true
                }
            }
        }

        func dims(_ d: [Int]) -> (k: Int, w: Int, K: Int, W: Int) {
            let cand: Set<Int> = [17, 26, 29]
            var kIdx: Int? = d.firstIndex(where: { cand.contains($0) })
            if kIdx == nil {
                kIdx = (d.count >= 2 && d[d.count - 2] <= d.last!) ? d.count - 2 : d.count - 1
            }
            var wIdx: Int? = (0..<d.count).reversed().first(where: { $0 != kIdx && d[$0] > 1 })
            if wIdx == nil { wIdx = d.count - 1 }
            let kk = kIdx ?? max(0, d.count - 2)
            let ww = wIdx ?? max(0, d.count - 1)
            return (kk, ww, d[kk], d[ww])
        }

        let ax = dims(dx), ay = dims(dy)
        let K = min(ax.K, ay.K), WX = ax.W, WY = ay.W

        enum Num { case f(UnsafePointer<Float>), d(UnsafePointer<Double>) }
        let px: Num = (X.dataType == .double) ? .d(X.dataPointer.bindMemory(to: Double.self, capacity: X.count))
                                              : .f(X.dataPointer.bindMemory(to: Float.self,  capacity: X.count))
        let py: Num = (Y.dataType == .double) ? .d(Y.dataPointer.bindMemory(to: Double.self, capacity: Y.count))
                                              : .f(Y.dataPointer.bindMemory(to: Float.self,  capacity: Y.count))
        @inline(__always) func get(_ n: Num, _ i: Int) -> Float { switch n { case .f(let p): return p[i]; case .d(let p): return Float(p[i]) } }

        func off(_ k: Int, _ i: Int, _ s: [Int], _ kIdx: Int, _ wIdx: Int) -> Int {
            return k * s[kIdx] + i * s[wIdx]
        }

        func decodeOnce(kx: Int, wx: Int, ky: Int, wy: Int) -> ([Int],[Int],[Float]) {
            var ix = Array(repeating: 0, count: K)
            var iy = Array(repeating: 0, count: K)
            var sc = Array(repeating: Float(0), count: K)
            for k in 0..<K {
                var mxX: Float = -.greatestFiniteMagnitude, argX = 0
                for i in 0..<WX {
                    let v = get(px, off(k, i, sx, kx, wx)); if v > mxX { mxX = v; argX = i }
                }
                var mxY: Float = -.greatestFiniteMagnitude, argY = 0
                for i in 0..<WY {
                    let v = get(py, off(k, i, sy, ky, wy)); if v > mxY { mxY = v; argY = i }
                }
                ix[k] = argX; iy[k] = argY; sc[k] = (mxX + mxY) * 0.5
            }
            return (ix, iy, sc)
        }

        let A = decodeOnce(kx: ax.k, wx: ax.w, ky: ay.k, wy: ay.w)
        let B = decodeOnce(kx: ax.w, wx: ax.k, ky: ay.w, wy: ay.k)

        func diversity(_ xs: [Int], _ ys: [Int]) -> Int {
            var set = Set<Int>(); set.reserveCapacity(xs.count)
            for i in 0..<xs.count { set.insert((xs[i] << 16) ^ ys[i]) }
            return set.count
        }
        let useA = diversity(A.0, A.1) >= diversity(B.0, B.1)
        let ix = useA ? A.0 : B.0
        let iy = useA ? A.1 : B.1
        let sc = useA ? A.2 : B.2

        // Map back to the full frame
        let swapAxes = (rotation == 90 || rotation == 270)
        let cropW = Double(cropRect.width), cropH = Double(cropRect.height)
        let rotW = swapAxes ? cropH : cropW, rotH = swapAxes ? cropW : cropH
        let scaleX = rotW / Double(inputWidth)
        let scaleY = rotH / Double(inputHeight)

        var out: [RTMPoseKeypoint] = []; out.reserveCapacity(K)
        for k in 0..<K {
            let xf = Double(ix[k]) / simccRatio
            let yf = Double(iy[k]) / simccRatio
            let xr = xf * scaleX
            let yr = yf * scaleY
            let p = rotatePoint(x: xr, y: yr, rotation: rotation, cropW: cropW, cropH: cropH)
            let Xv = min(Double(videoW),  max(0, Double(cropRect.minX) + p.x))
            let Yv = min(Double(videoH),  max(0, Double(cropRect.minY) + p.y))
            let S  = max(0, min(1, Double(sc[k])))
            out.append(RTMPoseKeypoint(x: Xv, y: Yv, score: S))
        }
        return out
    }

    private func rotatePoint(x: Double, y: Double, rotation: Int, cropW: Double, cropH: Double) -> CGPoint {
        let rot = normalizedRotation(rotation)
        if rot == 0 { return CGPoint(x: x, y: y) }

        let swap = (rot == 90 || rot == 270)
        let rotW = swap ? cropH : cropW
        let rotH = swap ? cropW : cropH
        let cx = cropW/2, cy = cropH/2
        let cxr = rotW/2, cyr = rotH/2

        let txp = x - cxr
        let typ = y - cyr

        let th = Double(rot) * Double.pi / 180.0
        let c = cos(th), s = sin(th)
        let tx = txp * c - typ * s
        let ty = txp * s + typ * c
        return CGPoint(x: tx + cx, y: ty + cy)
    }

    private func paddedRect(for box: YOLODetectionBox, videoW: Int, videoH: Int, pad: Double) -> CGRect {
        let cx = Double(box.x) + Double(box.w) / 2.0
        let cy = Double(box.y) + Double(box.h) / 2.0
        let pw = Double(box.w) * pad
        let ph = Double(box.h) * pad
        var minX = cx - pw/2, minY = cy - ph/2
        var maxX = cx + pw/2, maxY = cy + ph/2
        minX = max(0, minX); minY = max(0, minY)
        maxX = min(Double(videoW), maxX); maxY = min(Double(videoH), maxY)
        return CGRect(x: minX, y: minY, width: max(1, maxX - minX), height: max(1, maxY - minY))
    }

    private func renderPreview(baseImage: CGImage, keypoints: [RTMPoseKeypoint], outputURL: URL) throws -> URL {
        let w = baseImage.width, h = baseImage.height
        let img = UIGraphicsImageRenderer(size: CGSize(width: w, height: h)).image { ctx in
            let g = ctx.cgContext
            g.draw(baseImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            g.setLineWidth(4); g.setLineCap(.round)
            g.setStrokeColor(red: 0.2, green: 0.9, blue: 0.2, alpha: 0.9)
            g.setFillColor(red: 0.2, green: 0.9, blue: 0.2, alpha: 0.9)
            for (a,b) in skeletonPairs where a < keypoints.count && b < keypoints.count {
                let A = keypoints[a], B = keypoints[b]
                if A.score <= 0 || B.score <= 0 { continue }
                g.move(to: CGPoint(x: A.x, y: A.y)); g.addLine(to: CGPoint(x: B.x, y: B.y)); g.strokePath()
            }
            let r: CGFloat = 6
            for kp in keypoints where kp.score > 0 {
                g.fillEllipse(in: CGRect(x: kp.x - r/2, y: kp.y - r/2, width: r, height: r))
            }
        }
        guard let jpeg = img.jpegData(compressionQuality: 0.85) else { throw RTMPoseError.failedToCreatePreview }
        try jpeg.write(to: outputURL, options: .atomic)
        return outputURL
    }
}

// =====================================================
// MARK: – MotionBERT-Lite (2D → 3D lifting)
// =====================================================

enum MotionBERTError: Error {
    case poseJSONUnavailable
    case modelResourceMissing
    case modelUnavailable
    case predictionFailed
}

struct MotionBERTKeypoint3D: Codable {
    let X: Double
    let Y: Double
    let Z: Double
}

struct MotionBERTFrameOutput: Codable {
    let fi: Int
    let t: Double
    let ok: Bool
    let keypoints3d: [MotionBERTKeypoint3D]
}

struct MotionBERTTotals: Codable {
    let framesProcessed: Int
    let framesWith3D: Int
}

struct MotionBERTJSONOutput: Codable {
    struct Debug: Codable {
        let inputShape: [Int]
        let outputShape: [Int]
        let mapping: String
        let windowing: String
    }

    let videoWidth: Int
    let videoHeight: Int
    let fps: Double
    let sampledFps: Double
    let temporalWindow: Int
    let skeleton: String
    let frames: [MotionBERTFrameOutput]
    let totals: MotionBERTTotals
    let debug: Debug
}

struct MotionBERTSummary {
    let jsonURL: URL
    let previewURL: URL?
    let totals: MotionBERTTotals
}

final class MotionBERTAnalyzer {
    private let config: PipelineConfig
    private let motionConfig: PipelineConfig.MotionBERT
    private let debugConfig: PipelineConfig.Debug
    private let temporalWindow: Int
    private let windowCenter: Int
    private let stride: Int
    private struct Joint2D {
        let x: Double
        let y: Double
        let score: Double
    }

    private struct Frame2D {
        let index: Int
        let timestamp: Double
        let joints: [Joint2D]
    }

    private var model: MLModel?

    init(config: PipelineConfig) {
        self.config = config
        self.motionConfig = config.motionbert
        self.debugConfig = config.debug
        self.temporalWindow = max(1, config.motionbert.temporalWindow)
        self.windowCenter = min(max(config.motionbert.center, 0), max(0, self.temporalWindow - 1))
        self.stride = max(1, config.motionbert.stride)
    }

    func analyze(sessionDirectory: String, postprocessResult: Postprocess2DResult) throws -> MotionBERTSummary? {
        let normalized = postprocessResult.normalizedOutput

        NSLog("[MB] model computeUnits=NeuralEngine window=%d center=%d stride=%d skeleton=%@",
              temporalWindow, windowCenter, stride, motionConfig.skeleton)

        let frames2D = prepareFrames(from: normalized.frames)
        let frameCount = frames2D.count
        if frameCount == 0 {
            NSLog("[MB] No RTMPose frames found; skipping MotionBERT stage.")
            return nil
        }

        let sessionURL = URL(fileURLWithPath: sessionDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: sessionURL, withIntermediateDirectories: true)
        let outputURL = sessionURL.appendingPathComponent("motionbert_3d_keypoints.json")

        let mlmodelcURL = Bundle.main.url(forResource: "motionbert_lite_patched", withExtension: "mlmodelc")
        let mlpackageURL = Bundle.main.url(forResource: "motionbert_lite_patched", withExtension: "mlpackage")
        guard let modelURL = mlmodelcURL ?? mlpackageURL else {
            NSLog("[MB] ERROR model resource missing mlmodelc=%@ mlpackage=%@",
                  mlmodelcURL?.path ?? "nil", mlpackageURL?.path ?? "nil")
            throw MotionBERTError.modelResourceMissing
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        guard let motionModel = try? MLModel(contentsOf: modelURL, configuration: configuration) else {
            throw MotionBERTError.modelUnavailable
        }

        model = motionModel
        NSLog("[MB] I/O shapes input=[1,%d,17,3] output=[1,%d,17,3]", temporalWindow, temporalWindow)
        NSLog("[MB] Model loaded (computeUnits=NeuralEngine) — running…")

        var framesWith3D = 0
        var outputs: [MotionBERTFrameOutput] = []
        outputs.reserveCapacity(frameCount)

        if frameCount <= temporalWindow {
            let padLeft = (temporalWindow - frameCount) / 2
            let padRight = temporalWindow - frameCount - padLeft
            let first = frames2D.first!
            let last = frames2D.last!
            let padded = Array(repeating: first, count: padLeft) + frames2D + Array(repeating: last, count: padRight)
            guard let prediction = try? predict(window: padded) else {
                throw MotionBERTError.predictionFailed
            }

            for (index, frame) in frames2D.enumerated() {
                let timeIndex = padLeft + index
                let joints3d = extractFrame(from: prediction, timeIndex: timeIndex)
                outputs.append(MotionBERTFrameOutput(fi: frame.index, t: frame.timestamp, ok: true, keypoints3d: joints3d))
                if debugConfig.verboseLogging && every(frame.index, debugConfig.frameLogStride) {
                    NSLog("[MB] centerIdx=%d maps→ fi=%d t=%.3f", windowCenter, frame.index, frame.timestamp)
                }
            }
            framesWith3D = frameCount
        } else {
            var i = 0
            while i < frameCount {
                let prediction: MLMultiArray = try autoreleasepool {
                    let window = makeWindow(around: i, frames: frames2D)
                    return try self.predict(window: window)
                }
                let joints3d = self.extractFrame(from: prediction, timeIndex: windowCenter)
                outputs.append(MotionBERTFrameOutput(fi: frames2D[i].index, t: frames2D[i].timestamp, ok: true, keypoints3d: joints3d))
                if debugConfig.verboseLogging && every(i, debugConfig.frameLogStride) {
                    NSLog("[MB] centerIdx=%d maps→ fi=%d t=%.3f", windowCenter, frames2D[i].index, frames2D[i].timestamp)
                }
                framesWith3D += 1
                i += stride
            }
        }

        let totals = MotionBERTTotals(framesProcessed: frameCount, framesWith3D: framesWith3D)

        let jsonOutput = MotionBERTJSONOutput(
            videoWidth: normalized.videoWidth,
            videoHeight: normalized.videoHeight,
            fps: normalized.fps,
            sampledFps: normalized.sampledFps,
            temporalWindow: temporalWindow,
            skeleton: motionConfig.skeleton,
            frames: outputs,
            totals: totals,
            debug: .init(
                inputShape: [1, temporalWindow, 17, 3],
                outputShape: [1, temporalWindow, 17, 3],
                mapping: "postprocess_2d:h36m_normalized",
                windowing: "center=\(windowCenter), stride=\(stride)"
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(jsonOutput).write(to: outputURL, options: .atomic)

        NSLog("[MB] frames=%d with3D=%d -> motionbert_3d_keypoints.json", frameCount, framesWith3D)

        return MotionBERTSummary(jsonURL: outputURL, previewURL: nil, totals: totals)
    }

    func teardown() {
        model = nil
    }

    // MARK: – Helpers

    private func prepareFrames(from frames: [H36MNormalizedFrame]) -> [Frame2D] {
        var prepared: [Frame2D] = []
        prepared.reserveCapacity(frames.count)

        for frame in frames {
            let joints: [Joint2D] = frame.keypoints.map { values in
                let x = values.count > 0 ? values[0] : 0
                let y = values.count > 1 ? values[1] : 0
                let s = values.count > 2 ? values[2] : 0
                return Joint2D(x: x, y: y, score: s)
            }
            prepared.append(Frame2D(index: frame.fi, timestamp: frame.t, joints: joints))
        }

        return prepared
    }

    private func makeWindow(around index: Int, frames: [Frame2D]) -> [Frame2D] {
        var window: [Frame2D] = []
        window.reserveCapacity(temporalWindow)

        let total = frames.count
        for offset in 0..<temporalWindow {
            let idx = min(max(index - windowCenter + offset, 0), total - 1)
            window.append(frames[idx])
        }
        return window
    }

    private func predict(window: [Frame2D]) throws -> MLMultiArray {
        guard let model = model else { throw MotionBERTError.modelUnavailable }

        let shape = [1, temporalWindow, 17, 3].map { NSNumber(value: $0) }
        let input = try MLMultiArray(shape: shape, dataType: .float32)
        fill(input: input, with: window)

        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        let result = try model.prediction(from: provider)

        guard let output = result.featureValue(for: "output")?.multiArrayValue else {
            throw MotionBERTError.predictionFailed
        }
        return output
    }

    private func fill(input: MLMultiArray, with frames: [Frame2D]) {
        let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(input.dataPointer))
        var offset = 0
        for frame in frames {
            for joint in frame.joints {
                pointer[offset] = Float32(joint.x); offset += 1
                pointer[offset] = Float32(joint.y); offset += 1
                pointer[offset] = Float32(joint.score); offset += 1
            }
        }
    }

    private func extractFrame(from output: MLMultiArray, timeIndex: Int) -> [MotionBERTKeypoint3D] {
        let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(output.dataPointer))
        var joints: [MotionBERTKeypoint3D] = []
        joints.reserveCapacity(17)

        let baseOffset = timeIndex * 17 * 3
        for jointIndex in 0..<17 {
            let jointOffset = baseOffset + jointIndex * 3
            let X = Double(pointer[jointOffset])
            let Y = Double(pointer[jointOffset + 1])
            let Z = Double(pointer[jointOffset + 2])
            joints.append(MotionBERTKeypoint3D(X: X, Y: Y, Z: Z))
        }
        return joints
    }
}

// =====================================================
// MARK: – Pipeline wrapper
// =====================================================

enum VideoAnalysisStage: String { case yolo, rtmpose, motionbert }

struct VideoAnalysisStageError: Error, LocalizedError {
    let stage: VideoAnalysisStage
    let underlying: Error
    var errorDescription: String? {
        if let s = (underlying as NSError?)?.localizedDescription, !s.isEmpty { return s }
        return String(describing: underlying)
    }
}

struct VideoAnalysisResult {
    let yolo: YOLOAnalysisSummary
    let rtmpose: RTMPoseSummary
    let postprocess: Postprocess2DSummary
    let motionbert: MotionBERTSummary?
}

final class VideoAnalysisPipeline {
    private let config: PipelineConfig
    private let queue: DispatchQueue
    private let semaphore: DispatchSemaphore

    var debugConfig: PipelineConfig.Debug { config.debug }

    init(config: PipelineConfig) {
        self.config = config
        self.queue = DispatchQueue(
            label: config.runtime.dispatchQueueLabel,
            qos: .userInitiated,
            attributes: .concurrent
        )
        self.semaphore = DispatchSemaphore(value: max(1, config.runtime.maxConcurrentRuns))
    }

    func run(
        videoPath: String,
        sessionDirectory: String,
        sampledFps: Double,
        personStrategy: PersonSelectionStrategy = .bestScore,
        progress: ((String) -> Void)?,
        completion: @escaping (Result<VideoAnalysisResult, VideoAnalysisStageError>) -> Void
    ) {
        semaphore.wait()
        queue.async {
            defer { self.semaphore.signal() }
            let runID = String(UUID().uuidString.prefix(8))
            let startAll = CFAbsoluteTimeGetCurrent()
            NSLog("[PIPELINE %s] start video=%@ session=%@ maxConc=%d",
                  runID, redact(videoPath), sessionDirectory, self.config.runtime.maxConcurrentRuns)
            var currentStage: VideoAnalysisStage = .yolo
            do {
                progress?("Running YOLO…")
                var t0 = CFAbsoluteTimeGetCurrent()
                NSLog("[PIPELINE %s] YOLO → start", runID)
                let yoloSummary: YOLOAnalysisSummary = try autoreleasepool {
                    let a = VideoYOLOAnalyzer(config: self.config, personStrategy: personStrategy)
                    defer { a.teardown() }
                    return try a.analyze(videoAtPath: videoPath, sessionDirectory: sessionDirectory, sampledFps: sampledFps)
                }
                var t1 = CFAbsoluteTimeGetCurrent()
                NSLog("[PIPELINE %s] YOLO ← done frames=%d detections=%d (%.3fs)",
                      runID, yoloSummary.totals.framesProcessed, yoloSummary.totals.detections, t1 - t0)

                progress?("Running RTMPose…")
                t0 = CFAbsoluteTimeGetCurrent()
                NSLog("[PIPELINE %s] RTMPose → start", runID)
                let poseSummary: RTMPoseSummary = try autoreleasepool {
                    var analyzer: RTMPoseAnalyzer? = RTMPoseAnalyzer(config: self.config, personStrategy: personStrategy)
                    defer {
                        analyzer?.teardown()
                        analyzer = nil
                    }
                    return try analyzer!.analyze(videoAtPath: videoPath, sessionDirectory: sessionDirectory, yoloJSONURL: yoloSummary.jsonURL)
                }
                t1 = CFAbsoluteTimeGetCurrent()
                NSLog("[PIPELINE %s] RTMPose ← done frames=%d ok=%d (%.3fs)",
                      runID, poseSummary.totals.framesProcessed, poseSummary.totals.framesWithDetections, t1 - t0)
                currentStage = .rtmpose

                progress?("Refining 2D keypoints…")
                t0 = CFAbsoluteTimeGetCurrent()
                NSLog("[PIPELINE %s] postprocess_2d → start", runID)
                let postprocessResult: Postprocess2DResult = try autoreleasepool {
                    let processor = PosePostprocessor(
                        config: self.config.postprocess,
                        debug: self.config.debug,
                        headTopFactor: self.config.motionbert.headTopFactor,
                        skeletonName: self.config.motionbert.skeleton
                    )
                    let poseData = try Data(contentsOf: poseSummary.jsonURL)
                    let decoded = try JSONDecoder().decode(RTMPoseJSONOutput.self, from: poseData)
                    return try processor.run(poseOutput: decoded, sessionDirectory: sessionDirectory)
                }
                let postElapsed = CFAbsoluteTimeGetCurrent() - t0

                let postprocessSummary = Postprocess2DSummary(
                    refinedURL: postprocessResult.refinedJSONURL,
                    normalizedURL: postprocessResult.normalizedJSONURL,
                    totals: postprocessResult.refinedOutput.totals,
                    fps: postprocessResult.refinedOutput.fps,
                    sampledFps: postprocessResult.refinedOutput.sampledFps,
                    numKeypoints: postprocessResult.refinedOutput.numKeypoints
                )
                NSLog("[PIPELINE %s] postprocess_2d ← done refined=%@ normalized=%@ (%.3fs)",
                      runID,
                      postprocessSummary.refinedURL.lastPathComponent,
                      postprocessSummary.normalizedURL.lastPathComponent,
                      postElapsed)

                progress?("Running MotionBERT…")
                NSLog("[MotionBERT] Releasing RTMPose and freeing memory…")
                currentStage = .motionbert
                t0 = CFAbsoluteTimeGetCurrent()
                NSLog("[PIPELINE %s] MotionBERT → start", runID)
                let motionSummary: MotionBERTSummary? = try autoreleasepool {
                    var analyzer: MotionBERTAnalyzer? = MotionBERTAnalyzer(config: self.config)
                    defer {
                        analyzer?.teardown()
                        analyzer = nil
                    }
                    return try analyzer?.analyze(sessionDirectory: sessionDirectory, postprocessResult: postprocessResult)
                }
                if let mb = motionSummary {
                    let elapsed = CFAbsoluteTimeGetCurrent() - t0
                    NSLog("[PIPELINE %s] MotionBERT ← done frames=%d ok=%d (%.3fs)",
                          runID, mb.totals.framesProcessed, mb.totals.framesWith3D, elapsed)
                } else {
                    NSLog("[PIPELINE %s] MotionBERT ← skipped (no frames)", runID)
                }

                completion(.success(VideoAnalysisResult(yolo: yoloSummary, rtmpose: poseSummary, postprocess: postprocessSummary, motionbert: motionSummary)))
                NSLog("[PIPELINE %s] done (total=%.3fs)", runID, CFAbsoluteTimeGetCurrent() - startAll)
            } catch let e as VideoAnalysisStageError {
                NSLog("[PIPELINE %s] ERROR stage=%@ msg=%@", runID, e.stage.rawValue, e.localizedDescription)
                completion(.failure(e))
            } catch let e as YOLOAnalysisError {
                NSLog("[PIPELINE %s] ERROR stage=%@ msg=%@", runID, VideoAnalysisStage.yolo.rawValue, e.localizedDescription)
                completion(.failure(VideoAnalysisStageError(stage: .yolo, underlying: e)))
            } catch let e as RTMPoseError {
                NSLog("[PIPELINE %s] ERROR stage=%@ msg=%@", runID, VideoAnalysisStage.rtmpose.rawValue, e.localizedDescription)
                completion(.failure(VideoAnalysisStageError(stage: .rtmpose, underlying: e)))
            } catch let e as MotionBERTError {
                NSLog("[PIPELINE %s] ERROR stage=%@ msg=%@", runID, VideoAnalysisStage.motionbert.rawValue, e.localizedDescription)
                completion(.failure(VideoAnalysisStageError(stage: .motionbert, underlying: e)))
            } catch {
                NSLog("[PIPELINE %s] ERROR stage=%@ msg=%@", runID, currentStage.rawValue, error.localizedDescription)
                completion(.failure(VideoAnalysisStageError(stage: currentStage, underlying: error)))
            }
        }
    }
}

// =====================================================
// MARK: – Shared helpers
// =====================================================

private func normalizedRotation(_ deg: Int) -> Int {
    let n = ((deg % 360) + 360) % 360
    switch n {
    case 90, 180, 270:
        return n
    default:
        return 0
    }
}

private func rotateImage(_ image: CGImage, clockwiseDegrees deg: Int) -> CGImage? {
    let rot = normalizedRotation(deg)
    guard rot != 0 else { return image }

    let w = CGFloat(image.width)
    let h = CGFloat(image.height)
    let swapAxes = (rot == 90 || rot == 270)
    let size = swapAxes ? CGSize(width: h, height: w) : CGSize(width: w, height: h)

    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    let renderer = UIGraphicsImageRenderer(size: size, format: format)
    let img = renderer.image { ctx in
        let g = ctx.cgContext
        g.translateBy(x: size.width / 2, y: size.height / 2)
        g.rotate(by: CGFloat(Double(rot) * Double.pi / 180.0))
        g.translateBy(x: -w / 2, y: -h / 2)
        g.interpolationQuality = .high
        g.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    return img.cgImage
}

private func mapRectFromRotated(
    _ rect: CGRect,
    rotation: Int,
    originalWidth: Int,
    originalHeight: Int
) -> CGRect {
    let rot = normalizedRotation(rotation)
    guard originalWidth > 0, originalHeight > 0 else { return rect }
    if rot == 0 { return rect }

    let origW = Double(originalWidth)
    let origH = Double(originalHeight)
    let corners = [
        rect.origin,
        CGPoint(x: rect.maxX, y: rect.minY),
        CGPoint(x: rect.minX, y: rect.maxY),
        CGPoint(x: rect.maxX, y: rect.maxY)
    ]

    let mapped = corners.map { pointFromRotated($0, rotation: rot, originalWidth: origW, originalHeight: origH) }
    let xs = mapped.map { $0.x }
    let ys = mapped.map { $0.y }

    guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else {
        return rect
    }

    return CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))
}

private func pointFromRotated(
    _ point: CGPoint,
    rotation: Int,
    originalWidth: Double,
    originalHeight: Double
) -> CGPoint {
    switch rotation {
    case 90:
        return CGPoint(x: point.y, y: originalHeight - point.x)
    case 180:
        return CGPoint(x: originalWidth - point.x, y: originalHeight - point.y)
    case 270:
        return CGPoint(x: originalWidth - point.y, y: point.x)
    default:
        return point
    }
}

private func clampRectToBounds(_ rect: CGRect, width: Int, height: Int) -> CGRect {
    guard width > 0, height > 0 else { return .zero }
    let maxW = Double(width)
    let maxH = Double(height)

    let minX = min(max(0.0, rect.minX), maxW)
    let minY = min(max(0.0, rect.minY), maxH)
    let maxX = min(max(0.0, rect.maxX), maxW)
    let maxY = min(max(0.0, rect.maxY), maxH)

    return CGRect(x: minX, y: minY, width: max(0.0, maxX - minX), height: max(0.0, maxY - minY))
}

@inline(__always) func redact(_ path: String) -> String {
    return (path as NSString).lastPathComponent
}

@inline(__always) func every(_ i: Int, _ stride: Int) -> Bool {
    return stride > 0 ? (i % stride == 0) : true
}

func logKeysOnce(_ dict: [String: Any], tag: String, onlyOnce: inout Bool) {
    guard onlyOnce else { return }
    onlyOnce = false
    NSLog("[%@] output keys=%@", tag, dict.keys.sorted().joined(separator: ", "))
}
