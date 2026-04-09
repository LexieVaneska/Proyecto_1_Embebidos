pub const INPUT_VIDEO_PATH: &str = "videos/input_video5.mp4"; //Video de entrada
pub const OUTPUT_DIRECTORY: &str = "videos_salida"; //CArpeta para guardar el video procesado
pub const OUTPUT_VIDEO_PATH: &str = "videos_salida/video_procesado.mp4"; //Ruta del video final

pub const ORIGINAL_WINDOW_NAME: &str = "Video original"; //Nombre de la ventana del video original
pub const PROCESSED_WINDOW_NAME: &str = "Video procesado"; //Nombre de la ventana del video procesado

pub const FACE_CASCADE_PATH: &str =
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"; //Ruta para el modelo de detección de rostros
pub const FACEMARK_MODEL_PATH: &str = "models/lbfmodel.yaml"; //RUta para el modelo de landmarks faciales

pub const LEFT_EYE_INDICES: [usize; 6] = [36, 37, 38, 39, 40, 41]; //Números que corresponden al ojo izquierdo
pub const RIGHT_EYE_INDICES: [usize; 6] = [42, 43, 44, 45, 46, 47]; //Números que corresponden al ojo derecho

pub const EYE_CLOSED_EAR_THRESHOLD: f32 = 0.26; //Umbral para detectar que el ojo está cerrado
pub const EYE_BOX_HORIZONTAL_PADDING_FACTOR: f32 = 0.50; //Cuánto se ensancha la caja del ojo
pub const EYE_BOX_VERTICAL_PADDING_FACTOR: f32 = 1.20; //Cuanto se aumenta la altura del ojo
pub const MIN_EYE_BOX_PADDING: f32 = 6.0; //Margen mínimo aunque el ojo sea pequeño
//Tamaño del área donde se analizan el ojo
pub const EYE_ANALYSIS_WIDTH_FACTOR: f32 = 1.15; 
pub const EYE_ANALYSIS_HEIGHT_FACTOR: f32 = 0.50;
//Región central para enfocarse  en el análisis del ojo
pub const VISUAL_FOCUS_WIDTH_FACTOR: f32 = 0.72;
pub const VISUAL_FOCUS_HEIGHT_FACTOR: f32 = 0.55;
pub const VISUAL_DARK_ROW_MIN_RATIO: f64 = 0.10; //mínima proporción de oscuridad para considerar una fila relevante
pub const VISUAL_CLOSED_SPAN_RATIO_THRESHOLD: f32 = 0.18; //umbral de apertura visual mínima
pub const VISUAL_CLOSED_DARK_RATIO_THRESHOLD: f32 = 0.16; //umbral basado en oscuridad
pub const HYBRID_EAR_SOFT_THRESHOLD: f32 = 0.20; //umbral EAR más flexible
pub const EYELID_LINE_MIN_LENGTH_RATIO: f32 = 0.45; //tamaño mínimo de una línea de párpado detectada
pub const EYELID_LINE_MAX_VERTICAL_DRIFT: f32 = 0.18; //tolerancia vertical de esa línea
pub const EAR_HISTORY_MAX_POINTS: usize = 120; //Cuantos valores guarda
pub const EAR_GRAPH_WIDTH: i32 = 360; //Ancho de la gráfica
pub const EAR_GRAPH_HEIGHT: i32 = 130; //alto de la gráfica
pub const EAR_GRAPH_MARGIN: i32 = 20; //Separación respecto al borde del frame
//Rangos de la gráfica
pub const EAR_GRAPH_MIN_VALUE: f32 = 0.0; 
pub const EAR_GRAPH_MAX_VALUE: f32 = 0.35;
