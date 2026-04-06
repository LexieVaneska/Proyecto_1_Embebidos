pub const INPUT_VIDEO_PATH: &str = "videos/input_video4.mp4";
pub const OUTPUT_DIRECTORY: &str = "videos_salida";
pub const OUTPUT_VIDEO_PATH: &str = "videos_salida/video_procesado.mp4";

pub const ORIGINAL_WINDOW_NAME: &str = "Video original";
pub const PROCESSED_WINDOW_NAME: &str = "Video procesado";

pub const FACE_CASCADE_PATH: &str =
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
pub const FACEMARK_MODEL_PATH: &str = "models/lbfmodel.yaml";

pub const LEFT_EYE_INDICES: [usize; 6] = [36, 37, 38, 39, 40, 41];
pub const RIGHT_EYE_INDICES: [usize; 6] = [42, 43, 44, 45, 46, 47];

pub const EYE_CLOSED_EAR_THRESHOLD: f32 = 0.26;
pub const EYE_BOX_HORIZONTAL_PADDING_FACTOR: f32 = 0.50;
pub const EYE_BOX_VERTICAL_PADDING_FACTOR: f32 = 1.20;
pub const MIN_EYE_BOX_PADDING: f32 = 6.0;
pub const EYE_ANALYSIS_WIDTH_FACTOR: f32 = 1.15;
pub const EYE_ANALYSIS_HEIGHT_FACTOR: f32 = 0.50;
pub const VISUAL_FOCUS_WIDTH_FACTOR: f32 = 0.72;
pub const VISUAL_FOCUS_HEIGHT_FACTOR: f32 = 0.55;
pub const VISUAL_DARK_ROW_MIN_RATIO: f64 = 0.10;
pub const VISUAL_CLOSED_SPAN_RATIO_THRESHOLD: f32 = 0.18;
pub const VISUAL_CLOSED_DARK_RATIO_THRESHOLD: f32 = 0.16;
pub const HYBRID_EAR_SOFT_THRESHOLD: f32 = 0.20;
pub const EYELID_LINE_MIN_LENGTH_RATIO: f32 = 0.45;
pub const EYELID_LINE_MAX_VERTICAL_DRIFT: f32 = 0.18;
pub const EAR_HISTORY_MAX_POINTS: usize = 120;
pub const EAR_GRAPH_WIDTH: i32 = 360;
pub const EAR_GRAPH_HEIGHT: i32 = 130;
pub const EAR_GRAPH_MARGIN: i32 = 20;
pub const EAR_GRAPH_MIN_VALUE: f32 = 0.0;
pub const EAR_GRAPH_MAX_VALUE: f32 = 0.35;
