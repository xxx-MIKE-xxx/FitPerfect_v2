import Foundation

struct PipelineConfig: Decodable {
    struct Runtime: Decodable {
        let dispatchQueueLabel: String
        let maxConcurrentRuns: Int
    }

    struct YOLO: Decodable {
        let bboxMode: String
        let coordsOrigin: String
        let rotationCorrectionDeg: Int
        let channelOrder: String
        let imgsz: Int
        let stride: Int
        let padColor: [Int]
        let scoreThreshold: Double
        let nmsThreshold: Double
        let maxDetections: Int
    }

    struct RTMPose: Decodable {
        let rotationOverrideDeg: Int
        let padding: Double
        let inputWidth: Int
        let inputHeight: Int
        let simccRatio: Double
        let means: [Double]
        let stds: [Double]
        let minKeypointScore: Double
        let personSelection: String
        let channelOrder: String
    }

    struct Postprocess: Decodable {
        struct EMA: Decodable {
            let enabled: Bool
            let alpha: Double
        }

        struct Confidence: Decodable {
            let min: Double
            let floor: Double
        }

        struct GapFill: Decodable {
            let maxInterpolatedGap: Int
        }

        struct GlobalMotion: Decodable {
            let enabled: Bool
            let window: Int
        }

        struct Normalize: Decodable {
            let enabled: Bool
            let rootJoint: Int
            let scaleJointA: Int
            let scaleJointB: Int
            let epsilon: Double
        }

        let ema: EMA
        let confidence: Confidence
        let gapFill: GapFill
        let globalMotion: GlobalMotion
        let normalize: Normalize
    }

    struct MotionBERT: Decodable {
        let temporalWindow: Int
        let center: Int
        let stride: Int
        let headTopFactor: Double
        let skeleton: String
    }

    let runtime: Runtime
    let yolo: YOLO
    let rtmpose: RTMPose
    let postprocess: Postprocess
    let motionbert: MotionBERT
}

extension PipelineConfig {
    static func load(from resource: String = "pipeline_config", bundle: Bundle = .main) throws -> PipelineConfig {
        guard let url = bundle.url(forResource: resource, withExtension: "json") else {
            throw NSError(domain: "PipelineConfig", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing pipeline_config.json in bundle"])
        }

        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(PipelineConfig.self, from: data)
    }
}
