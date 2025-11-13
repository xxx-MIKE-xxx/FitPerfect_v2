import Foundation

struct PipelineConfig: Decodable {
    struct Debug: Decodable {
        let verboseLogging: Bool
        let frameLogStride: Int
        let dumpModelOutputShapesOnce: Bool

        init(
            verboseLogging: Bool = false,
            frameLogStride: Int = 10,
            dumpModelOutputShapesOnce: Bool = true
        ) {
            self.verboseLogging = verboseLogging
            self.frameLogStride = frameLogStride
            self.dumpModelOutputShapesOnce = dumpModelOutputShapesOnce
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.verboseLogging = try container.decodeIfPresent(Bool.self, forKey: .verboseLogging) ?? false
            self.frameLogStride = try container.decodeIfPresent(Int.self, forKey: .frameLogStride) ?? 10
            self.dumpModelOutputShapesOnce = try container.decodeIfPresent(Bool.self, forKey: .dumpModelOutputShapesOnce) ?? true
        }

        private enum CodingKeys: String, CodingKey {
            case verboseLogging
            case frameLogStride
            case dumpModelOutputShapesOnce
        }
    }
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
    let debug: Debug

    private enum CodingKeys: String, CodingKey {
        case runtime
        case yolo
        case rtmpose
        case postprocess
        case motionbert
        case debug
    }

    init(
        runtime: Runtime,
        yolo: YOLO,
        rtmpose: RTMPose,
        postprocess: Postprocess,
        motionbert: MotionBERT,
        debug: Debug = Debug()
    ) {
        self.runtime = runtime
        self.yolo = yolo
        self.rtmpose = rtmpose
        self.postprocess = postprocess
        self.motionbert = motionbert
        self.debug = debug
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.runtime = try container.decode(Runtime.self, forKey: .runtime)
        self.yolo = try container.decode(YOLO.self, forKey: .yolo)
        self.rtmpose = try container.decode(RTMPose.self, forKey: .rtmpose)
        self.postprocess = try container.decode(Postprocess.self, forKey: .postprocess)
        self.motionbert = try container.decode(MotionBERT.self, forKey: .motionbert)
        self.debug = try container.decodeIfPresent(Debug.self, forKey: .debug) ?? Debug()
    }
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
