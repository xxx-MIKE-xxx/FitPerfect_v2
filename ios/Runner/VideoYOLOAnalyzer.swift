import Foundation
import AVFoundation
import CoreML
import Vision

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
    private let queue = DispatchQueue(label: "com.fitperfect.yolo", qos: .userInitiated)

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

    func analyze(videoAtPath videoPath: String, sessionDirectory: String, sampledFps: Double, completion: @escaping (Result<YOLOAnalysisSummary, Error>) -> Void) {
        queue.async {
            do {
                let videoURL = URL(fileURLWithPath: videoPath)
                let asset = AVAsset(url: videoURL)
                guard let track = asset.tracks(withMediaType: .video).first else {
                    throw YOLOAnalysisError.videoTrackUnavailable
                }

                let configuration = MLModelConfiguration()
                configuration.computeUnits = .all
                guard let yolo = try? YOLOv3Tiny(configuration: configuration) else {
                    throw YOLOAnalysisError.modelUnavailable
                }
                let visionModel = try VNCoreMLModel(for: yolo.model)

                let transformedSize = track.naturalSize.applying(track.preferredTransform)
                let videoWidth = Int(abs(transformedSize.width))
                let videoHeight = Int(abs(transformedSize.height))

                let fps = track.nominalFrameRate > 0 ? Double(track.nominalFrameRate) : self.estimateFrameRate(for: track)
                let durationSeconds = CMTimeGetSeconds(asset.duration)

                let generator = AVAssetImageGenerator(asset: asset)
                generator.appliesPreferredTrackTransform = true
                generator.requestedTimeToleranceAfter = .zero
                generator.requestedTimeToleranceBefore = .zero

                let effectiveSampledFps = max(sampledFps, 1.0)
                let samplingStep = 1.0 / effectiveSampledFps
                var frames: [YOLODetectionFrame] = []
                var totalDetections = 0

                var currentTime = 0.0
                while currentTime < durationSeconds {
                    autoreleasepool {
                        let time = CMTime(seconds: currentTime, preferredTimescale: asset.duration.timescale)
                        if let cgImage = try? generator.copyCGImage(at: time, actualTime: nil) {
                            let frame = self.processFrame(
                                cgImage: cgImage,
                                timestamp: currentTime,
                                model: visionModel,
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
                let data = try encoder.encode(output)
                try data.write(to: jsonURL, options: .atomic)

                let summary = YOLOAnalysisSummary(jsonURL: jsonURL, totals: totals)
                completion(.success(summary))
            } catch {
                completion(.failure(error))
            }
        }
    }

    private func processFrame(
        cgImage: CGImage,
        timestamp: Double,
        model: VNCoreMLModel,
        videoWidth: Int,
        videoHeight: Int
    ) -> YOLODetectionFrame {
        let request = VNCoreMLRequest(model: model)
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
