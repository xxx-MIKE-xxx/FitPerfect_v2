import Foundation
import simd

struct PoseRefinedKeypoint: Codable {
    let x: Double
    let y: Double
    let score: Double
}

struct PoseRefinedFrame: Codable {
    let fi: Int
    let t: Double
    let ok: Bool
    let keypoints: [PoseRefinedKeypoint]
}

struct PoseRefinedTotals: Codable {
    let framesProcessed: Int
    let framesWithDetections: Int
}

struct PoseRefinedJSONOutput: Codable {
    let videoWidth: Int
    let videoHeight: Int
    let fps: Double
    let sampledFps: Double
    let numKeypoints: Int
    let frames: [PoseRefinedFrame]
    let totals: PoseRefinedTotals
}

struct H36MNormalizedFrame: Codable {
    let fi: Int
    let t: Double
    let ok: Bool
    let keypoints: [[Double]]
}

struct H36MNormalizedJSONOutput: Codable {
    let videoWidth: Int
    let videoHeight: Int
    let fps: Double
    let sampledFps: Double
    let skeleton: String
    let frames: [H36MNormalizedFrame]
}

struct Postprocess2DResult {
    let refinedJSONURL: URL
    let normalizedJSONURL: URL
    let refinedOutput: PoseRefinedJSONOutput
    let normalizedOutput: H36MNormalizedJSONOutput
}

struct Postprocess2DSummary {
    let refinedURL: URL
    let normalizedURL: URL
    let totals: PoseRefinedTotals
    let fps: Double
    let sampledFps: Double
    let numKeypoints: Int
}

final class PosePostprocessor {
    private let config: PipelineConfig.Postprocess
    private let debug: PipelineConfig.Debug
    private let headTopFactor: Double
    private let skeletonName: String

    init(config: PipelineConfig.Postprocess, debug: PipelineConfig.Debug, headTopFactor: Double, skeletonName: String) {
        self.config = config
        self.debug = debug
        self.headTopFactor = headTopFactor
        self.skeletonName = skeletonName
    }

    func run(
        poseOutput: RTMPoseJSONOutput,
        sessionDirectory: String
    ) throws -> Postprocess2DResult {
        let jointCount = poseOutput.numKeypoints
        guard jointCount > 0 else {
            throw NSError(domain: "PosePostprocessor", code: -1, userInfo: [NSLocalizedDescriptionKey: "RTMPose output has no keypoints"])
        }

        let normMode = config.normalize.enabled ? "enabled" : "disabled"
        NSLog("[POST2D] refine interp=gap<=%d ma_win=%d score_thresh=%.3f norm=%@ bbox_ema=%.2f",
              config.gapFill.maxInterpolatedGap,
              config.globalMotion.window,
              config.confidence.min,
              normMode,
              config.ema.alpha)

        struct JointSample {
            var point: SIMD3<Double>
            var valid: Bool
        }

        let frameCount = poseOutput.frames.count
        var frames: [[JointSample]] = Array(repeating: Array(repeating: JointSample(point: SIMD3<Double>(repeating: 0), valid: false), count: jointCount), count: frameCount)

        for (index, frame) in poseOutput.frames.enumerated() {
            for (jointIndex, kp) in frame.keypoints.enumerated() where jointIndex < jointCount {
                let clampedScore = max(config.confidence.floor, kp.score)
                frames[index][jointIndex] = JointSample(point: SIMD3<Double>(kp.x, kp.y, clampedScore), valid: kp.score >= config.confidence.min && frame.ok)
            }
        }

        var totalGapsFilled = 0
        var jointGapCounts = Array(repeating: 0, count: jointCount)
        var firstMeta: (center: SIMD2<Double>, scale: Double)?
        var lastMeta: (center: SIMD2<Double>, scale: Double)?

        // Gap filling via linear interpolation up to configured window
        if config.gapFill.maxInterpolatedGap > 0 {
            for joint in 0..<jointCount {
                var startIndex: Int? = nil
                var startValue: SIMD3<Double>? = nil
                var idx = 0
                while idx < frameCount {
                    if frames[idx][joint].valid {
                        startIndex = idx
                        startValue = frames[idx][joint].point
                        idx += 1
                        continue
                    }

                    let gapStart = idx
                    while idx < frameCount && !frames[idx][joint].valid { idx += 1 }
                    let gapEnd = idx

                    if let startIdx = startIndex, let startVal = startValue, gapEnd < frameCount, frames[gapEnd][joint].valid {
                        let gapLength = gapEnd - gapStart
                        if gapLength <= config.gapFill.maxInterpolatedGap {
                            let endVal = frames[gapEnd][joint].point
                            for step in 0..<gapLength {
                                let alpha = Double(step + 1) / Double(gapLength + 1)
                                let interpolated = startVal + (endVal - startVal) * alpha
                                frames[gapStart + step][joint].point = interpolated
                                frames[gapStart + step][joint].valid = true
                                totalGapsFilled += 1
                                jointGapCounts[joint] += 1
                            }
                        }
                    }

                    startIndex = gapEnd < frameCount && frames[gapEnd][joint].valid ? gapEnd : startIndex
                    startValue = gapEnd < frameCount && frames[gapEnd][joint].valid ? frames[gapEnd][joint].point : startValue
                }
            }
        }

        // Optional EMA smoothing per joint
        if config.ema.enabled {
            let alpha = config.ema.alpha
            let oneMinusAlpha = 1.0 - alpha
            for joint in 0..<jointCount {
                var previous: SIMD3<Double>? = nil
                for frameIndex in 0..<frameCount {
                    var sample = frames[frameIndex][joint]
                    if let prev = previous {
                        sample.point = prev * oneMinusAlpha + sample.point * alpha
                    }
                    if sample.valid {
                        previous = sample.point
                    }
                    frames[frameIndex][joint] = sample
                }
            }
        }

        // Optional global motion smoothing: simple moving average on root joint
        if config.globalMotion.enabled {
            let window = max(1, config.globalMotion.window)
            if window > 1 {
                let halfWindow = window / 2
                for joint in 0..<jointCount {
                    var buffer: [SIMD3<Double>] = []
                    buffer.reserveCapacity(window)
                    for frameIndex in 0..<frameCount {
                        buffer.append(frames[frameIndex][joint].point)
                        if buffer.count > window { buffer.removeFirst() }
                        let aggregated = buffer.reduce(SIMD3<Double>(repeating: 0), +) / Double(buffer.count)
                        frames[frameIndex][joint].point = aggregated
                    }
                }
            }
        }

        // Prepare refined frames and convert to H36M layout
        var refinedFrames: [PoseRefinedFrame] = []
        refinedFrames.reserveCapacity(frameCount)

        var h36mFrames: [H36MNormalizedFrame] = []
        h36mFrames.reserveCapacity(frameCount)

        for (index, frame) in poseOutput.frames.enumerated() {
            let jointSamples = frames[index]
            var refinedKeypoints: [PoseRefinedKeypoint] = []
            refinedKeypoints.reserveCapacity(jointCount)
            var hasValid = false
            for sample in jointSamples {
                refinedKeypoints.append(PoseRefinedKeypoint(x: sample.point.x, y: sample.point.y, score: min(max(sample.point.z, 0.0), 1.0)))
                hasValid = hasValid || sample.valid
            }

            refinedFrames.append(
                PoseRefinedFrame(
                    fi: frame.fi,
                    t: frame.t,
                    ok: frame.ok || hasValid,
                    keypoints: refinedKeypoints
                )
            )

            let h36mJoints = convertToH36M(coco: refinedKeypoints)
            let normalization = normalize(h36mJoints: h36mJoints)
            if firstMeta == nil {
                firstMeta = (normalization.center, normalization.scale)
            }
            lastMeta = (normalization.center, normalization.scale)
            if debug.verboseLogging && every(frame.fi, debug.frameLogStride) {
                NSLog("[POST2D] fi=%d center=(%.1f,%.1f) scale=%.3f",
                      frame.fi, normalization.center.x, normalization.center.y, normalization.scale)
            }
            h36mFrames.append(
                H36MNormalizedFrame(
                    fi: frame.fi,
                    t: frame.t,
                    ok: frame.ok || hasValid,
                    keypoints: normalization.values
                )
            )
        }

        let refinedTotals = PoseRefinedTotals(
            framesProcessed: refinedFrames.count,
            framesWithDetections: refinedFrames.filter { $0.ok }.count
        )

        let refinedOutput = PoseRefinedJSONOutput(
            videoWidth: poseOutput.videoWidth,
            videoHeight: poseOutput.videoHeight,
            fps: poseOutput.fps,
            sampledFps: poseOutput.sampledFps,
            numKeypoints: jointCount,
            frames: refinedFrames,
            totals: refinedTotals
        )

        let normalizedOutput = H36MNormalizedJSONOutput(
            videoWidth: poseOutput.videoWidth,
            videoHeight: poseOutput.videoHeight,
            fps: poseOutput.fps,
            sampledFps: poseOutput.sampledFps,
            skeleton: skeletonName,
            frames: h36mFrames
        )

        NSLog("[POST2D] frames=%d refined=%d normalized=%d", frameCount, refinedFrames.count, h36mFrames.count)

        let sessionURL = URL(fileURLWithPath: sessionDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: sessionURL, withIntermediateDirectories: true)

        let refinedURL = sessionURL.appendingPathComponent("rtmpose_keypoints_refined.json")
        let normalizedURL = sessionURL.appendingPathComponent("h36m_2d_normalized.json")

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let refinedData = try encoder.encode(refinedOutput)
        try refinedData.write(to: refinedURL, options: .atomic)

        let normalizedData = try encoder.encode(normalizedOutput)
        try normalizedData.write(to: normalizedURL, options: .atomic)

        NSLog("[POST2D] gapsFilled total=%d byJoint=%@", totalGapsFilled, compactJointGapHistogram(jointGapCounts))
        if let firstMeta = firstMeta, let lastMeta = lastMeta {
            NSLog("[POST2D] centers first=(%.1f,%.1f) last=(%.1f,%.1f) scale first=%.3f last=%.3f",
                  firstMeta.center.x, firstMeta.center.y,
                  lastMeta.center.x, lastMeta.center.y,
                  firstMeta.scale, lastMeta.scale)
        }

        return Postprocess2DResult(
            refinedJSONURL: refinedURL,
            normalizedJSONURL: normalizedURL,
            refinedOutput: refinedOutput,
            normalizedOutput: normalizedOutput
        )
    }

    private func convertToH36M(coco: [PoseRefinedKeypoint]) -> [SIMD3<Double>] {
        guard coco.count >= 17 else { return Array(repeating: SIMD3<Double>(repeating: 0), count: 17) }

        func midpoint(_ a: PoseRefinedKeypoint, _ b: PoseRefinedKeypoint) -> PoseRefinedKeypoint {
            PoseRefinedKeypoint(
                x: (a.x + b.x) * 0.5,
                y: (a.y + b.y) * 0.5,
                score: min(a.score, b.score)
            )
        }

        let lHip = coco[11]
        let rHip = coco[12]
        let pelvis = midpoint(lHip, rHip)
        let neck = midpoint(coco[5], coco[6])
        let torso = midpoint(pelvis, neck)
        let nose = coco[0]
        let factor = headTopFactor
        let headTop = PoseRefinedKeypoint(
            x: nose.x + factor * (nose.x - neck.x),
            y: nose.y + factor * (nose.y - neck.y),
            score: nose.score
        )

        let joints: [PoseRefinedKeypoint] = [
            pelvis,
            rHip,
            coco[14],
            coco[16],
            lHip,
            coco[13],
            coco[15],
            torso,
            neck,
            nose,
            headTop,
            coco[5],
            coco[7],
            coco[9],
            coco[6],
            coco[8],
            coco[10]
        ]

        return joints.map { SIMD3<Double>($0.x, $0.y, $0.score) }
    }

    private func compactJointGapHistogram(_ counts: [Int]) -> String {
        let nonZero = counts.enumerated().filter { $0.element > 0 }
        guard !nonZero.isEmpty else { return "{}" }
        let parts = nonZero.map { "j\($0.offset)=\($0.element)" }
        return "{\(parts.joined(separator: ","))}"
    }

    private func normalize(h36mJoints: [SIMD3<Double>]) -> (values: [[Double]], center: SIMD2<Double>, scale: Double) {
        guard !h36mJoints.isEmpty else { return ([], SIMD2<Double>(repeating: 0), 1.0) }

        let rootIdx = max(0, min(config.normalize.rootJoint, h36mJoints.count - 1))
        let scaleA = max(0, min(config.normalize.scaleJointA, h36mJoints.count - 1))
        let scaleB = max(0, min(config.normalize.scaleJointB, h36mJoints.count - 1))
        let root = h36mJoints[rootIdx]
        let center = SIMD2<Double>(root.x, root.y)
        let bone = simd_length(h36mJoints[scaleA] - h36mJoints[scaleB])
        let denom = max(bone, config.normalize.epsilon)

        guard config.normalize.enabled, h36mJoints.count >= 17 else {
            return (h36mJoints.map { [$0.x, $0.y, $0.z] }, center, denom)
        }

        let normalized = h36mJoints.map { joint -> [Double] in
            let centered = joint - root
            return [centered.x / denom, centered.y / denom, joint.z]
        }
        return (normalized, center, denom)
    }
}
