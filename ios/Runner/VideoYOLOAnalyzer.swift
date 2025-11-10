import Foundation
import AVFoundation
import CoreML
import Vision
import UIKit

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

struct YOLODetectionFrame: Codable {
    let t: Double
    let boxes: [YOLODetectionBox]
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

final class VideoYOLOAnalyzer {
    private var asset: AVAsset?
    private var track: AVAssetTrack?
    private var model: YOLOv3Tiny?
    private var visionModel: VNCoreMLModel?
    private var generator: AVAssetImageGenerator?

    private let classLabels: [String] = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    func analyze(
        videoAtPath videoPath: String,
        sessionDirectory: String,
        sampledFps: Double
    ) throws -> YOLOAnalysisSummary {
        let videoURL = URL(fileURLWithPath: videoPath)
        let asset = AVAsset(url: videoURL)
        self.asset = asset

        guard let track = asset.tracks(withMediaType: .video).first else {
            throw YOLOAnalysisError.videoTrackUnavailable
        }
        self.track = track

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        guard let yoloModel = try? YOLOv3Tiny(configuration: configuration) else {
            throw YOLOAnalysisError.modelUnavailable
        }
        model = yoloModel
        visionModel = try VNCoreMLModel(for: yoloModel.model)

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

        let effectiveSampledFps = max(sampledFps, 1.0)
        let samplingStep = 1.0 / effectiveSampledFps

        var frames: [YOLODetectionFrame] = []
        var totalDetections = 0

        var currentTime = 0.0
        let timeScale = asset.duration.timescale != 0 ? asset.duration.timescale : 600

        while currentTime < durationSeconds {
            autoreleasepool {
                let time = CMTime(seconds: currentTime, preferredTimescale: timeScale)
                if let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) {
                    let frame = self.processFrame(
                        cgImage: cgImage,
                        timestamp: currentTime,
                        videoWidth: videoWidth,
                        videoHeight: videoHeight
                    )
                    totalDetections += frame.boxes.count
                    frames.append(frame)
                } else {
                    let emptyFrame = YOLODetectionFrame(
                        t: round(currentTime * 100) / 100,
                        boxes: []
                    )
                    frames.append(emptyFrame)
                }
            }
            currentTime += samplingStep
        }

        let totals = YOLODetectionTotals(
            framesProcessed: frames.count,
            detections: totalDetections
        )

        let output = YOLODetectionOutput(
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            fps: fps,
            sampledFps: effectiveSampledFps,
            frames: frames,
            totals: totals
        )

        let sessionURL = URL(fileURLWithPath: sessionDirectory, isDirectory: true)
        let jsonURL = sessionURL.appendingPathComponent("yolo_detections.json")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(output)
            try data.write(to: jsonURL, options: .atomic)
        } catch {
            throw YOLOAnalysisError.failedToWriteJSON
        }

        return YOLOAnalysisSummary(jsonURL: jsonURL, totals: totals)
    }

    func teardown() {
        generator = nil
        visionModel = nil
        model = nil
        track = nil
        asset = nil
    }

    private func processFrame(
        cgImage: CGImage,
        timestamp: Double,
        videoWidth: Int,
        videoHeight: Int
    ) -> YOLODetectionFrame {
        guard let visionModel = visionModel else {
            return YOLODetectionFrame(t: round(timestamp * 100) / 100, boxes: [])
        }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        do {
            try handler.perform([request])
            guard let observations = request.results as? [VNRecognizedObjectObservation] else {
                return YOLODetectionFrame(t: round(timestamp * 100) / 100, boxes: [])
            }

            var boxes: [YOLODetectionBox] = []
            for observation in observations {
                guard let topLabel = observation.labels.first else { continue }
                let label = topLabel.identifier
                let clsIndex = classLabels.firstIndex(of: label) ?? -1
                let rect = observation.boundingBox
                let width = Double(videoWidth)
                let height = Double(videoHeight)

                let x = Int((rect.minX * width).rounded())
                let y = Int(((1 - rect.maxY) * height).rounded())
                let w = Int((rect.width * width).rounded())
                let h = Int((rect.height * height).rounded())

                let box = YOLODetectionBox(
                    cls: clsIndex,
                    label: label,
                    score: Double(topLabel.confidence),
                    x: x,
                    y: y,
                    w: w,
                    h: h
                )
                boxes.append(box)
            }

            return YOLODetectionFrame(
                t: round(timestamp * 100) / 100,
                boxes: boxes
            )
        } catch {
            return YOLODetectionFrame(t: round(timestamp * 100) / 100, boxes: [])
        }
    }

    private func estimateFrameRate(for track: AVAssetTrack) -> Double {
        if track.nominalFrameRate > 0 {
            return Double(track.nominalFrameRate)
        }
        let minFrameDuration = track.minFrameDuration
        if minFrameDuration.isValid && minFrameDuration.value != 0 {
            return Double(minFrameDuration.timescale) / Double(minFrameDuration.value)
        }
        return 0
    }
}

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

    let videoWidth: Int
    let videoHeight: Int
    let fps: Double
    let sampledFps: Double
    let numKeypoints: Int
    let simccRatio: Double
    let inputSize: InputSize
    let frames: [RTMPoseFrame]
    let totals: RTMPoseTotals
}

struct RTMPoseSummary {
    let jsonURL: URL
    let previewURL: URL?
    let totals: RTMPoseTotals
    let numKeypoints: Int
}

// === Replace the whole RTMPoseAnalyzer with this version ===
final class RTMPoseAnalyzer {
    private let paddingFactor: Double
    private let personStrategy: PersonSelectionStrategy

    // Use the auto-generated wrapper, not a raw MLModel
    private var model: RTMPose?
    private var asset: AVAsset?
    private var generator: AVAssetImageGenerator?

    private let inputWidth = 192
    private let inputHeight = 256
    private let simccRatio: Double = 2.0
    private let channelMeans: [Float] = [123.675, 116.28, 103.53]
    private let channelStds: [Float] = [58.395, 57.12, 57.375]

    private let skeletonPairs: [(Int, Int)] = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16), (5, 1), (6, 2), (1, 3), (2, 4)
    ]

    init(paddingFactor: Double = 1.25, personStrategy: PersonSelectionStrategy = .bestScore) {
        self.paddingFactor = paddingFactor
        self.personStrategy = personStrategy
    }

    func analyze(
        videoAtPath videoPath: String,
        sessionDirectory: String,
        yoloJSONURL: URL
    ) throws -> RTMPoseSummary {
        // Load YOLO JSON as before
        let jsonData = try Data(contentsOf: yoloJSONURL)
        let yoloOutput = try JSONDecoder().decode(YOLODetectionOutput.self, from: jsonData)

        // Video + model setup
        let videoURL = URL(fileURLWithPath: videoPath)
        let asset = AVAsset(url: videoURL)
        self.asset = asset

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU   // more compatible than .all for this model
        // Use generated wrapper
        self.model = try RTMPose(configuration: configuration)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        self.generator = generator

        var frames: [RTMPoseFrame] = []
        var framesWithDetections = 0
        var previewURL: URL?
        var numKeypointsDetected = 0
        let sessionURL = URL(fileURLWithPath: sessionDirectory, isDirectory: true)
        let poseJSONURL = sessionURL.appendingPathComponent("rtmpose_keypoints.json")
        let previewOutputURL = sessionURL.appendingPathComponent("rtmpose_preview.jpg")

        let duration = asset.duration
        let timeScale = duration.timescale != 0 ? duration.timescale : 600

        for frame in yoloOutput.frames {
            autoreleasepool {
                guard let selectedBox = self.selectPersonBox(from: frame.boxes) else {
                    frames.append(RTMPoseFrame(t: frame.t, ok: false, keypoints: []))
                    return
                }

                let time = CMTime(seconds: frame.t, preferredTimescale: timeScale)
                guard let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) else {
                    frames.append(RTMPoseFrame(t: frame.t, ok: false, keypoints: []))
                    return
                }

                guard
                    let prepared = self.prepareInput(
                        from: cgImage,
                        box: selectedBox,
                        videoWidth: yoloOutput.videoWidth,
                        videoHeight: yoloOutput.videoHeight
                    ),
                    let prediction = try? self.predictPose(from: prepared.array)
                else {
                    frames.append(RTMPoseFrame(t: frame.t, ok: false, keypoints: []))
                    return
                }

                let keypoints = self.decode(
                    prediction: prediction,
                    cropRect: prepared.cropRect,
                    videoWidth: yoloOutput.videoWidth,
                    videoHeight: yoloOutput.videoHeight
                )

                guard !keypoints.isEmpty else {
                    frames.append(RTMPoseFrame(t: frame.t, ok: false, keypoints: []))
                    return
                }

                // Track the maximum number of keypoints detected on any frame
                numKeypointsDetected = max(numKeypointsDetected, keypoints.count)

                frames.append(RTMPoseFrame(t: frame.t, ok: true, keypoints: keypoints))
                framesWithDetections += 1

                if previewURL == nil, let p = try? self.renderPreview(
                    baseImage: cgImage,
                    keypoints: keypoints,
                    outputURL: previewOutputURL
                ) {
                    previewURL = p
                }
            }
        }


        let totals = RTMPoseTotals(
            framesProcessed: frames.count,
            framesWithDetections: framesWithDetections
        )

        let output = RTMPoseJSONOutput(
            videoWidth: yoloOutput.videoWidth,
            videoHeight: yoloOutput.videoHeight,
            fps: yoloOutput.fps,
            sampledFps: yoloOutput.sampledFps,
            numKeypoints: numKeypointsDetected,
            simccRatio: simccRatio,
            inputSize: .init(w: inputWidth, h: inputHeight),
            frames: frames,
            totals: totals
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(output)
        try data.write(to: poseJSONURL, options: .atomic)

        return RTMPoseSummary(
            jsonURL: poseJSONURL,
            previewURL: previewURL,
            totals: totals,
            numKeypoints: numKeypointsDetected
        )
    }

    func teardown() {
        generator = nil
        asset = nil
        model = nil
    }

    private func selectPersonBox(from boxes: [YOLODetectionBox]) -> YOLODetectionBox? {
        let personBoxes = boxes.filter { $0.label == "person" }
        guard !personBoxes.isEmpty else { return nil }
        switch personStrategy {
        case .bestScore: return personBoxes.max(by: { $0.score < $1.score })
        case .largest:   return personBoxes.max(by: { ($0.w * $0.h) < ($1.w * $1.h) })
        }
    }

    private func prepareInput(
        from image: CGImage,
        box: YOLODetectionBox,
        videoWidth: Int,
        videoHeight: Int
    ) -> (array: MLMultiArray, cropRect: CGRect)? {
        let paddedRect = paddedBoundingBox(
            for: box,
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            paddingFactor: paddingFactor
        )

        let imageHeight = image.height
        let cropRect = CGRect(
            x: paddedRect.minX, y: paddedRect.minY,
            width: paddedRect.width, height: paddedRect.height
        )

        let cgCropRect = CGRect(
            x: cropRect.minX,
            y: Double(imageHeight) - cropRect.minY - cropRect.height,
            width: cropRect.width,
            height: cropRect.height
        )

        guard let cropped = image.cropping(to: cgCropRect.integral) else { return nil }

        var pixels = [UInt8](repeating: 0, count: inputWidth * inputHeight * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = inputWidth * 4
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

        guard let context = CGContext(
            data: &pixels, width: inputWidth, height: inputHeight,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cropped, in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight))

        let pixelCount = inputWidth * inputHeight
        var floatData = [Float](repeating: 0, count: pixelCount * 3)
        for idx in 0..<pixelCount {
            let base = idx * 4
            let r = Float(pixels[base + 0])
            let g = Float(pixels[base + 1])
            let b = Float(pixels[base + 2])
            floatData[idx] = (r - channelMeans[0]) / channelStds[0]
            floatData[pixelCount + idx] = (g - channelMeans[1]) / channelStds[1]
            floatData[(2 * pixelCount) + idx] = (b - channelMeans[2]) / channelStds[2]
        }

        let shape: [NSNumber] = [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)]
        guard let array = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }
        floatData.withUnsafeBytes { buf in
            if let base = buf.baseAddress {
                memcpy(array.dataPointer, base, floatData.count * MemoryLayout<Float>.size)
            }
        }
        return (array, cropRect)
    }

    // Use the generated input/output types
    private func predictPose(from array: MLMultiArray) throws -> RTMPoseOutput {
        guard let model = model else { throw RTMPoseError.modelUnavailable }
        let input = RTMPoseInput(input: array)
        return try model.prediction(input: input)
    }

    private func decode(
        prediction: RTMPoseOutput,
        cropRect: CGRect,
        videoWidth: Int,
        videoHeight: Int
    ) -> [RTMPoseKeypoint] {

        let simccX = prediction.simcc_x
        let simccY = prediction.simcc_y

        // Shapes & strides
        let dx = simccX.shape.map { Int(truncating: $0) }   // e.g. [1,17,384] or [1,384,17]
        let dy = simccY.shape.map { Int(truncating: $0) }
        let sx = simccX.strides.map { Int(truncating: $0) } // e.g. [17*384, 384, 1] if [B,K,W]
        let sy = simccY.strides.map { Int(truncating: $0) }

        // Find which dim is Keypoints (K) and which is the SIMCC length (W)
        func dims(_ d: [Int]) -> (kIdx: Int, wIdx: Int, K: Int, W: Int) {
            // Prefer a known K (17/26/29); otherwise pick the smaller of the last two dims.
            let candidates: Set<Int> = [17, 26, 29]
            let kIdx = (d.indices.first { candidates.contains(d[$0]) } ?? ((d[1] < d[2]) ? 1 : 2))
            let wIdx = (kIdx == 1) ? 2 : 1
            return (kIdx, wIdx, d[kIdx], d[wIdx])
        }

        let (kIdxX, wIdxX, Kx, Wx) = dims(dx)
        let (kIdxY, wIdxY, Ky, Wy) = dims(dy)
        let K = min(Kx, Ky)            // safety if they differ slightly
        let WX = Wx, WY = Wy

        // Raw pointers
        let px = simccX.dataPointer.bindMemory(to: Float.self, capacity: simccX.count)
        let py = simccY.dataPointer.bindMemory(to: Float.self, capacity: simccY.count)

        // Stride helpers to compute flat offsets
        func offX(_ k: Int, _ i: Int) -> Int { k * sx[kIdxX] + i * sx[wIdxX] }
        func offY(_ k: Int, _ i: Int) -> Int { k * sy[kIdxY] + i * sy[wIdxY] }

        var keypoints: [RTMPoseKeypoint] = []
        keypoints.reserveCapacity(K)

        let scaleX = cropRect.width  / Double(inputWidth)
        let scaleY = cropRect.height / Double(inputHeight)

        for k in 0..<K {
            var maxX: Float = -Float.greatestFiniteMagnitude, idxX = 0
            for i in 0..<WX {
                let v = px[offX(k, i)]
                if v > maxX { maxX = v; idxX = i }
            }

            var maxY: Float = -Float.greatestFiniteMagnitude, idxY = 0
            for i in 0..<WY {
                let v = py[offY(k, i)]
                if v > maxY { maxY = v; idxY = i }
            }

            let xIn = Double(idxX) / simccRatio
            let yIn = Double(idxY) / simccRatio

            let mappedX = min(Double(videoWidth),  max(0, cropRect.minX + xIn * scaleX))
            let mappedY = min(Double(videoHeight), max(0, cropRect.minY + yIn * scaleY))

            let score = max(0, min(1, Double((maxX + maxY) / 2.0)))
            keypoints.append(RTMPoseKeypoint(x: mappedX, y: mappedY, score: score))
        }

        return keypoints
    }


    private func extractFloats(from multiArray: MLMultiArray) -> [Float]? {
        let count = multiArray.count
        switch multiArray.dataType {
        case .float32:
            let p = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: p, count: count))
        case .double:
            let p = multiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
            return (0..<count).map { Float(p[$0]) }
        default:
            return nil
        }
    }

    private func renderPreview(
        baseImage: CGImage,
        keypoints: [RTMPoseKeypoint],
        outputURL: URL
    ) throws -> URL {
        let width = baseImage.width, height = baseImage.height
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let image = renderer.image { ctx in
            let cg = ctx.cgContext
            cg.draw(baseImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            cg.setLineWidth(4); cg.setLineCap(.round)
            cg.setStrokeColor(red: 0.2, green: 0.9, blue: 0.2, alpha: 0.9)
            cg.setFillColor(red: 0.2, green: 0.9, blue: 0.2, alpha: 0.9)
            for (aIdx, bIdx) in skeletonPairs where aIdx < keypoints.count && bIdx < keypoints.count {
                let a = keypoints[aIdx], b = keypoints[bIdx]
                if a.score <= 0 || b.score <= 0 { continue }
                cg.move(to: CGPoint(x: a.x, y: a.y)); cg.addLine(to: CGPoint(x: b.x, y: b.y)); cg.strokePath()
            }
            let r: CGFloat = 6
            for kp in keypoints where kp.score > 0 {
                cg.fillEllipse(in: CGRect(x: kp.x - r/2, y: kp.y - r/2, width: r, height: r))
            }
        }
        guard let jpeg = image.jpegData(compressionQuality: 0.85) else { throw RTMPoseError.failedToCreatePreview }
        try jpeg.write(to: outputURL, options: .atomic)
        return outputURL
    }

    private func paddedBoundingBox(
        for box: YOLODetectionBox,
        videoWidth: Int,
        videoHeight: Int,
        paddingFactor: Double
    ) -> CGRect {
        let cx = Double(box.x) + Double(box.w) / 2.0
        let cy = Double(box.y) + Double(box.h) / 2.0
        let pw = Double(box.w) * paddingFactor
        let ph = Double(box.h) * paddingFactor
        var minX = cx - pw/2, minY = cy - ph/2
        var maxX = cx + pw/2, maxY = cy + ph/2
        minX = max(0, minX); minY = max(0, minY)
        maxX = min(Double(videoWidth), maxX); maxY = min(Double(videoHeight), maxY)
        return CGRect(x: minX, y: minY, width: max(1, maxX - minX), height: max(1, maxY - minY))
    }
}


enum VideoAnalysisStage: String {
    case yolo
    case rtmpose
}

struct VideoAnalysisStageError: Error, LocalizedError {
    let stage: VideoAnalysisStage
    let underlying: Error

    var errorDescription: String? {
        if let localized = (underlying as NSError?)?.localizedDescription, !localized.isEmpty {
            return localized
        }
        return String(describing: underlying)
    }
}

struct VideoAnalysisResult {
    let yolo: YOLOAnalysisSummary
    let rtmpose: RTMPoseSummary
}

final class VideoAnalysisPipeline {
    private let queue = DispatchQueue(label: "com.fitperfect.analysis", qos: .userInitiated)

    func run(
        videoPath: String,
        sessionDirectory: String,
        sampledFps: Double,
        personStrategy: PersonSelectionStrategy = .bestScore,
        progress: ((String) -> Void)?,
        completion: @escaping (Result<VideoAnalysisResult, VideoAnalysisStageError>) -> Void
    ) {
        queue.async {
            do {
                progress?("Running YOLO…")
                let yoloSummary: YOLOAnalysisSummary = try autoreleasepool {
                    let analyzer = VideoYOLOAnalyzer()
                    defer { analyzer.teardown() }
                    return try analyzer.analyze(
                        videoAtPath: videoPath,
                        sessionDirectory: sessionDirectory,
                        sampledFps: sampledFps
                    )
                }
                NSLog("YOLO stage complete. Frames: %d, detections: %d", yoloSummary.totals.framesProcessed, yoloSummary.totals.detections)

                progress?("Running RTMPose…")
                let poseSummary: RTMPoseSummary = try autoreleasepool {
                    let analyzer = RTMPoseAnalyzer(personStrategy: personStrategy)
                    defer { analyzer.teardown() }
                    return try analyzer.analyze(
                        videoAtPath: videoPath,
                        sessionDirectory: sessionDirectory,
                        yoloJSONURL: yoloSummary.jsonURL
                    )
                }
                NSLog("RTMPose stage complete. Frames: %d, detections: %d", poseSummary.totals.framesProcessed, poseSummary.totals.framesWithDetections)

                let result = VideoAnalysisResult(yolo: yoloSummary, rtmpose: poseSummary)
                completion(.success(result))
            } catch let stageError as VideoAnalysisStageError {
                completion(.failure(stageError))
            } catch let error as YOLOAnalysisError {
                completion(.failure(VideoAnalysisStageError(stage: .yolo, underlying: error)))
            } catch let error as RTMPoseError {
                completion(.failure(VideoAnalysisStageError(stage: .rtmpose, underlying: error)))
            } catch {
                completion(.failure(VideoAnalysisStageError(stage: .yolo, underlying: error)))
            }
        }
    }
}
