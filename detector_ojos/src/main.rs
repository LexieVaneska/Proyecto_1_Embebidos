use std::collections::VecDeque;
use std::fs; //crear carpetas y manejar archivos
use std::path::Path;

use opencv::{ 
    core::{self, Mat, Point, Point2f, Rect, Scalar, Size, Vec4i, Vector}, //Mat para imágenes, Point para el centro, Point2f para landmarks, Rect para bounding boxes, Scalar para colores, Size para dimensiones
    face, //face para usar FacemarkLBF y dibujar landmarks
    highgui, //highgui para mostrar ventanas y manejar eventos de teclado
    imgproc, //imgproc para convertir a gris y dibujar rectángulos
    objdetect, //objdetect para usar los clasificadores Haar Cascade
    prelude::*, //prelude para traer a scope varias funciones y tipos comunes de OpenCV
    videoio, //videoio para leer y escribir videos, manejar cámaras 
};

// indica dónde se espera encontrar el video de entrada
const INPUT_VIDEO_PATH: &str = "videos/input_video4.mp4";

// define la carpeta donde guardaremos los videos procesados
const OUTPUT_DIRECTORY: &str = "videos_salida";

// nombre del archivo de salida
const OUTPUT_VIDEO_PATH: &str = "videos_salida/video_procesado.mp4";

// nombres de las ventanas para vista original y procesada
const ORIGINAL_WINDOW_NAME: &str = "Video original";
const PROCESSED_WINDOW_NAME: &str = "Video procesado";

// ruta del clasificador que detecta rostros frontales
const FACE_CASCADE_PATH: &str = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

// ruta del modelo preentrenado de FacemarkLBF
const FACEMARK_MODEL_PATH: &str = "models/lbfmodel.yaml";

// índices de landmarks para el ojo izquierdo en un modelo facial de 68 puntos
const LEFT_EYE_INDICES: [usize; 6] = [36, 37, 38, 39, 40, 41];

// índices de landmarks para el ojo derecho en un modelo facial de 68 puntos
const RIGHT_EYE_INDICES: [usize; 6] = [42, 43, 44, 45, 46, 47];

// umbral aproximado del Eye Aspect Ratio para considerar un ojo cerrado
const EYE_CLOSED_EAR_THRESHOLD: f32 = 0.26;
const EYE_BOX_HORIZONTAL_PADDING_FACTOR: f32 = 0.50;
const EYE_BOX_VERTICAL_PADDING_FACTOR: f32 = 1.20;
const MIN_EYE_BOX_PADDING: f32 = 6.0;
const EYE_ANALYSIS_WIDTH_FACTOR: f32 = 1.15;
const EYE_ANALYSIS_HEIGHT_FACTOR: f32 = 0.50;
const VISUAL_FOCUS_WIDTH_FACTOR: f32 = 0.72;
const VISUAL_FOCUS_HEIGHT_FACTOR: f32 = 0.55;
const VISUAL_DARK_ROW_MIN_RATIO: f64 = 0.10;
const VISUAL_CLOSED_SPAN_RATIO_THRESHOLD: f32 = 0.18;
const VISUAL_CLOSED_DARK_RATIO_THRESHOLD: f32 = 0.16;
const HYBRID_EAR_SOFT_THRESHOLD: f32 = 0.20;
const EYELID_LINE_MIN_LENGTH_RATIO: f32 = 0.45;
const EYELID_LINE_MAX_VERTICAL_DRIFT: f32 = 0.18;
const EAR_HISTORY_MAX_POINTS: usize = 120;
const EAR_GRAPH_WIDTH: i32 = 360;
const EAR_GRAPH_HEIGHT: i32 = 130;
const EAR_GRAPH_MARGIN: i32 = 20;
const EAR_GRAPH_MIN_VALUE: f32 = 0.0;
const EAR_GRAPH_MAX_VALUE: f32 = 0.35;

fn main() -> opencv::Result<()> { // función principal
    // se crea la carpeta de salida si todavía no existe

    ensure_output_directory_exists()?;

    // abrir el video de entrada definido en la constante
    let mut capture = open_input_video(INPUT_VIDEO_PATH)?;
    let fps = capture.get(videoio::CAP_PROP_FPS)?;
    let frame_width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    // crea el objeto que escribirá el video de salida con las mismas propiedades que el video original
    let mut writer = create_output_writer(OUTPUT_VIDEO_PATH, fps, frame_width, frame_height)?;

    // se cargan los detectores una sola vez al inicio para no hacerlo en cada frame
    let mut face_detector = load_cascade_classifier(FACE_CASCADE_PATH)?;
    let mut facemark = load_facemark_model(FACEMARK_MODEL_PATH)?;

    // se calcula el retraso entre frames usando el FPS real del video
    let frame_delay = calculate_frame_delay(fps);

    // Creamos ventanas separadas para el video original y el procesado
    highgui::named_window(ORIGINAL_WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(ORIGINAL_WINDOW_NAME, 960, 540)?;
    highgui::named_window(PROCESSED_WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(PROCESSED_WINDOW_NAME, 960, 540)?;

    // se muestra por consola los datos principales del video
    println!("Video de entrada: {}", INPUT_VIDEO_PATH);
    println!("Resolucion detectada: {}x{}", frame_width, frame_height);
    println!("FPS detectados: {:.2}", fps);
    println!("Video de salida: {}", OUTPUT_VIDEO_PATH);
    println!("Presiona la tecla 'q' para cerrar la ventana.");

    // contador que ayuda a saber cuántos frames se han procesado
    let mut processed_frames = 0;
    let mut ear_history = VecDeque::<f32>::new();

    loop { //ciclo infinito: seguir leyendo frames hasta que el video se acabe o el usuario salga
        
        let mut frame = Mat::default();

        // OpenCV lee el siguiente frame del archivo
        let was_frame_read = capture.read(&mut frame)?;

        // si ya no había más frames o si el frame leído está vacío, se sale del ciclo
        if !was_frame_read || frame.empty() {
            break;
        }


        // etapa de procesamiento:
        // 1. detectar varios candidatos de rostro
        // 2. escoger solo el mejor rostro
        // 3. usar FacemarkLBF para obtener landmarks del rostro
        // 4. dibujar el rostro y los puntos faciales sobre la imagen
        let mut processing_result = process_frame(&frame, &mut face_detector, &mut facemark)?;
        update_ear_history(&mut ear_history, processing_result.average_ear);
        draw_ear_graph(&mut processing_result.frame, &ear_history)?;

        // muestra el frame original y el procesado en ventanas separadas
        highgui::imshow(ORIGINAL_WINDOW_NAME, &frame)?;
        highgui::imshow(PROCESSED_WINDOW_NAME, &processing_result.frame)?;

        // guarda el frame dentro del vídeo de salida
        writer.write(&processing_result.frame)?;

        processed_frames += 1; //suma uno al contador

        // espera el tiempo aproximado de un frame según el FPS del video
        // si se presiona q, el programa termina
        let pressed_key = highgui::wait_key(frame_delay)?;
        if pressed_key == i32::from(b'q') {
            break;
        }
    }

    println!("Frames procesados: {}", processed_frames);
    println!("Procesamiento finalizado."); 

    Ok(()) // indica que todo salió bien
}

fn ensure_output_directory_exists() -> opencv::Result<()> { //función auxiliar para crear la carpeta de salida si no existe
    //revisa que existe
    if Path::new(OUTPUT_DIRECTORY).exists() {
        return Ok(());
    }

    // si no existe, intenta crearla. Si falla, devuelve un error con mensaje
    fs::create_dir_all(OUTPUT_DIRECTORY).map_err(|error| {
        opencv::Error::new(
            0,
            format!(
                "No se pudo crear la carpeta de salida '{}': {}",
                OUTPUT_DIRECTORY, error
            ),
        )
    })?;

    Ok(())
}
//función que abre el vídeo de entrada
fn open_input_video(video_path: &str) -> opencv::Result<videoio::VideoCapture> { 
    //abre el archivo de video usando opencv
    let capture = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;

    // se revisa que el video se abrió correctamente
    if !capture.is_opened()? {
        return Err(opencv::Error::new(
            0,
            format!(
                "No se pudo abrir el video de entrada '{}'. Coloca el archivo dentro de detector_ojos/ o ajusta la ruta.",
                video_path
            ),
        ));
    }

    Ok(capture)
}

fn load_cascade_classifier(cascade_path: &str) -> opencv::Result<objdetect::CascadeClassifier> {
    // revisa si el archivo XML del clasificador existe antes de cargarlo
    if !Path::new(cascade_path).exists() {
        return Err(opencv::Error::new(
            0,
            format!("No se encontró el clasificador Haar Cascade en '{}'", cascade_path),
        ));
    }

    // carga el clasificador desde el archivo XML
    let classifier = objdetect::CascadeClassifier::new(cascade_path)?;
    Ok(classifier)
}

fn load_facemark_model(
    model_path: &str,
) -> opencv::Result<opencv::core::Ptr<face::FacemarkLBF>> {
    // revisa si el archivo del modelo existe antes de intentar cargarlo
    if !Path::new(model_path).exists() {
        return Err(opencv::Error::new(
            0,
            format!("No se encontró el modelo de Facemark en '{}'", model_path),
        ));
    }

    // crea una instancia del detector de landmarks con parámetros por defecto
    let mut facemark = face::FacemarkLBF::create_def()?;

    // carga el modelo entrenado desde disco
    facemark.load_model(model_path)?;

    Ok(facemark)
}

fn calculate_frame_delay(fps: f64) -> i32 {
    // convierte FPS a milisegundos por frame para que la vista previa se vea a velocidad parecida al video real
    if fps > 0.0 {
        (1000.0 / fps).round() as i32
    } else {
        33
    }
}

fn create_output_writer( //función que construye el escritor del vídeo de salida
    output_path: &str,
    fps: f64,
    frame_width: i32,
    frame_height: i32,
) -> opencv::Result<videoio::VideoWriter> {
    // se define el codec de video 
    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;

    // si el video reporta FPS inválidos, se usa 30 como valor por defecto  
    let output_fps = if fps > 0.0 { fps } else { 30.0 };

    // se crea el escritor de video con las propiedades del original
    let writer = videoio::VideoWriter::new(
        output_path,
        fourcc,
        output_fps,
        Size::new(frame_width, frame_height),
        true,
    )?;

    Ok(writer)
}

fn process_frame(
    frame: &Mat,
    face_detector: &mut objdetect::CascadeClassifier,
    facemark: &mut opencv::core::Ptr<face::FacemarkLBF>,
) -> opencv::Result<FrameProcessingResult> { //función que procesa cada frame
    // crea un frame de salida para dibujar sobre él sin alterar el frame original
    let mut output_frame = Mat::default();
    frame.copy_to(&mut output_frame)?;
    let mut average_ear = None;

    // OpenCV detecta mejor con imágenes en escala de grises
    let mut gray_frame = Mat::default();
    imgproc::cvt_color(frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;

    // esta normalización mejora un poco el contraste antes de detectar
    let mut equalized_frame = Mat::default();
    imgproc::equalize_hist(&gray_frame, &mut equalized_frame)?;

    // aquí se guardarán todos los rostros detectados en el frame
    let mut faces = Vector::<Rect>::new();
    face_detector.detect_multi_scale(
        &equalized_frame,
        &mut faces,
        1.1,
        5,
        objdetect::CASCADE_SCALE_IMAGE,
        Size::new(80, 80),
        Size::new(0, 0),
    )?;

    // se escoge un único rostro, dando preferencia a un rostro grande y cercano al centro
    if let Some(face) = select_primary_face(&faces, frame.size()?) {
        draw_face_bounding_box(&mut output_frame, face)?;

        // colocamos el rostro en un vector porque Facemark::fit espera una lista de caras
        let mut selected_faces = Vector::<Rect>::new();
        selected_faces.push(face);

        // aquí se guardarán los landmarks detectados para cada rostro
        let mut landmarks = Vector::<Vector<Point2f>>::new();
        let landmarks_found = facemark.fit(frame, &selected_faces, &mut landmarks)?;

        // si el modelo logró ajustar puntos faciales, los dibujamos sobre la imagen
        if landmarks_found && !landmarks.is_empty() {
            draw_landmarks(&mut output_frame, &landmarks)?;

            // además de dibujar todos los puntos, obtenemos las cajas y el estado de ambos ojos
            average_ear = draw_eye_boxes_and_labels_from_landmarks(
                &mut output_frame,
                &equalized_frame,
                &landmarks,
            )?;
        }
    }

    draw_visual_legend(&mut output_frame)?;

    Ok(FrameProcessingResult {
        frame: output_frame,
        average_ear,
    })
}

struct FrameProcessingResult {
    frame: Mat,
    average_ear: Option<f32>,
}

fn select_primary_face(faces: &Vector<Rect>, frame_size: Size) -> Option<Rect> {
    // si no hay rostros detectados, devolvemos None
    if faces.is_empty() {
        return None;
    }

    // calculamos el centro del frame para favorecer rostros cercanos a esa zona
    let frame_center = Point::new(frame_size.width / 2, frame_size.height / 2);

    // guardamos el mejor rostro encontrado hasta el momento y su puntaje
    let mut best_face = None;
    let mut best_score = f64::MIN;

    // evaluamos cada rectángulo detectado
    for face in faces.iter() {
        let face_area = f64::from(face.width * face.height);
        let face_center = Point::new(face.x + face.width / 2, face.y + face.height / 2);

        // medimos qué tan lejos está el rostro del centro de la imagen
        let distance_x = f64::from(face_center.x - frame_center.x);
        let distance_y = f64::from(face_center.y - frame_center.y);
        let distance_to_center = (distance_x * distance_x + distance_y * distance_y).sqrt();

        // un puntaje alto favorece rostros grandes y penaliza los que están lejos del centro
        let score = face_area - (distance_to_center * 250.0);

        if score > best_score {
            best_score = score;
            best_face = Some(face);
        }
    }

    best_face
}

fn draw_face_bounding_box(frame: &mut Mat, face: Rect) -> opencv::Result<()> {
    // verde para rostro
    imgproc::rectangle(
        frame,
        face,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    Ok(())
}

fn draw_landmarks(frame: &mut Mat, landmarks: &Vector<Vector<Point2f>>) -> opencv::Result<()> {
    // dibuja todos los puntos faciales detectados para cada rostro
    for facial_points in landmarks.iter() {
        face::draw_facemarks(frame, &facial_points, Scalar::new(0.0, 0.0, 255.0, 0.0))?;
    }
    Ok(())
}

fn draw_eye_boxes_and_labels_from_landmarks(
    frame: &mut Mat,
    gray_frame: &Mat,
    landmarks: &Vector<Vector<Point2f>>,
) -> opencv::Result<Option<f32>> {
    let mut detected_ears = Vec::<f32>::new();

    // recorremos los landmarks de cada rostro detectado
    for facial_points in landmarks.iter() {
        if facial_points.len() < 48 {
            continue;
        }

        let mut left_eye_detection = None;
        let mut right_eye_detection = None;

        if let Some(left_eye_box) = build_eye_box_from_indices(&facial_points, &LEFT_EYE_INDICES) {
            let left_eye_ear = compute_eye_aspect_ratio(&facial_points, &LEFT_EYE_INDICES);
            detected_ears.push(left_eye_ear);
            let left_eye_analysis_box =
                build_eye_analysis_box_from_corners(&facial_points, &LEFT_EYE_INDICES);
            let left_eye_visual_metrics =
                compute_eye_visual_metrics(gray_frame, left_eye_analysis_box)?;
            let left_eye_is_closed = is_eye_closed(left_eye_ear, left_eye_visual_metrics);
            let left_eye_status = get_eye_status_label(
                left_eye_is_closed,
                "Ojo izquierdo",
            );
            left_eye_detection = Some((
                left_eye_box,
                left_eye_status,
                left_eye_ear,
                left_eye_visual_metrics,
                left_eye_is_closed,
            ));
        }

        if let Some(right_eye_box) = build_eye_box_from_indices(&facial_points, &RIGHT_EYE_INDICES) {
            let right_eye_ear = compute_eye_aspect_ratio(&facial_points, &RIGHT_EYE_INDICES);
            detected_ears.push(right_eye_ear);
            let right_eye_analysis_box =
                build_eye_analysis_box_from_corners(&facial_points, &RIGHT_EYE_INDICES);
            let right_eye_visual_metrics =
                compute_eye_visual_metrics(gray_frame, right_eye_analysis_box)?;
            let right_eye_is_closed = is_eye_closed(right_eye_ear, right_eye_visual_metrics);
            let right_eye_status = get_eye_status_label(
                right_eye_is_closed,
                "Ojo derecho",
            );
            right_eye_detection = Some((
                right_eye_box,
                right_eye_status,
                right_eye_ear,
                right_eye_visual_metrics,
                right_eye_is_closed,
            ));
        }

        if let (Some(left_eye), Some(right_eye)) = (&left_eye_detection, &right_eye_detection) {
            if left_eye.4 && right_eye.4 {
                draw_closed_eye_fill(frame, left_eye.0)?;
                draw_closed_eye_fill(frame, right_eye.0)?;
            }
        }

        if let Some((left_eye_box, left_eye_status, left_eye_ear, left_eye_visual_metrics, _)) =
            left_eye_detection
        {
            draw_eye_bounding_box(frame, left_eye_box)?;
            draw_eye_status_label(
                frame,
                left_eye_box,
                &left_eye_status,
                left_eye_ear,
                left_eye_visual_metrics,
            )?;
        }

        if let Some((
            right_eye_box,
            right_eye_status,
            right_eye_ear,
            right_eye_visual_metrics,
            _,
        )) = right_eye_detection
        {
            draw_eye_bounding_box(frame, right_eye_box)?;
            draw_eye_status_label(
                frame,
                right_eye_box,
                &right_eye_status,
                right_eye_ear,
                right_eye_visual_metrics,
            )?;
        }
    }

    if detected_ears.is_empty() {
        Ok(None)
    } else {
        let ear_sum: f32 = detected_ears.iter().copied().sum();
        Ok(Some(ear_sum / detected_ears.len() as f32))
    }
}

fn update_ear_history(ear_history: &mut VecDeque<f32>, average_ear: Option<f32>) {
    if let Some(ear) = average_ear {
        ear_history.push_back(ear);
        while ear_history.len() > EAR_HISTORY_MAX_POINTS {
            ear_history.pop_front();
        }
    }
}

fn draw_ear_graph(frame: &mut Mat, ear_history: &VecDeque<f32>) -> opencv::Result<()> {
    if ear_history.is_empty() {
        return Ok(());
    }

    let frame_size = frame.size()?;
    let graph_x = frame_size.width - EAR_GRAPH_WIDTH - EAR_GRAPH_MARGIN;
    let graph_y = frame_size.height - EAR_GRAPH_HEIGHT - EAR_GRAPH_MARGIN;
    if graph_x < 0 || graph_y < 0 {
        return Ok(());
    }

    let graph_rect = Rect::new(graph_x, graph_y, EAR_GRAPH_WIDTH, EAR_GRAPH_HEIGHT);
    imgproc::rectangle(
        frame,
        graph_rect,
        Scalar::new(20.0, 20.0, 20.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::rectangle(
        frame,
        graph_rect,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    draw_threshold_line(frame, graph_rect)?;
    draw_ear_curve(frame, graph_rect, ear_history)?;
    draw_graph_text(
        frame,
        "EAR promedio",
        Point::new(graph_rect.x + 10, graph_rect.y + 18),
    )?;

    if let Some(last_ear) = ear_history.back() {
        draw_graph_text(
            frame,
            &format!("Actual: {:.3}", last_ear),
            Point::new(graph_rect.x + 10, graph_rect.y + 38),
        )?;
    }

    Ok(())
}

fn draw_threshold_line(frame: &mut Mat, graph_rect: Rect) -> opencv::Result<()> {
    let threshold_y = map_ear_to_graph_y(EYE_CLOSED_EAR_THRESHOLD, graph_rect);
    imgproc::line(
        frame,
        Point::new(graph_rect.x, threshold_y),
        Point::new(graph_rect.x + graph_rect.width - 1, threshold_y),
        Scalar::new(0.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        0,
    )?;
    Ok(())
}

fn draw_ear_curve(frame: &mut Mat, graph_rect: Rect, ear_history: &VecDeque<f32>) -> opencv::Result<()> {
    if ear_history.len() < 2 {
        return Ok(());
    }

    let step_x = (graph_rect.width - 1) as f32 / (EAR_HISTORY_MAX_POINTS.saturating_sub(1)) as f32;
    let mut previous_point = None;

    for (index, ear) in ear_history.iter().enumerate() {
        let x = graph_rect.x + ((index as f32) * step_x).round() as i32;
        let y = map_ear_to_graph_y(*ear, graph_rect);
        let current_point = Point::new(x, y);

        if let Some(prev_point) = previous_point {
            imgproc::line(
                frame,
                prev_point,
                current_point,
                Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }

        previous_point = Some(current_point);
    }

    Ok(())
}

fn map_ear_to_graph_y(ear: f32, graph_rect: Rect) -> i32 {
    let clamped_ear = ear.clamp(EAR_GRAPH_MIN_VALUE, EAR_GRAPH_MAX_VALUE);
    let normalized = (clamped_ear - EAR_GRAPH_MIN_VALUE) / (EAR_GRAPH_MAX_VALUE - EAR_GRAPH_MIN_VALUE);
    let inverted = 1.0 - normalized;
    graph_rect.y + (inverted * (graph_rect.height - 1) as f32).round() as i32
}

fn draw_graph_text(frame: &mut Mat, text: &str, origin: Point) -> opencv::Result<()> {
    imgproc::put_text(
        frame,
        text,
        origin,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.45,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn build_eye_box_from_indices(
    facial_points: &Vector<Point2f>,
    eye_indices: &[usize],
) -> Option<Rect> {
    // iniciamos límites extremos para calcular el rectángulo mínimo que contiene el ojo
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    // recorremos únicamente los puntos que pertenecen al ojo
    for &index in eye_indices {
        let point = facial_points.get(index).ok()?;
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }

    // agregamos un margen más amplio para que la caja del ojo sea más visible
    let eye_width = max_x - min_x;
    let eye_height = max_y - min_y;
    let horizontal_padding =
        (eye_width * EYE_BOX_HORIZONTAL_PADDING_FACTOR).max(MIN_EYE_BOX_PADDING);
    let vertical_padding =
        (eye_height * EYE_BOX_VERTICAL_PADDING_FACTOR).max(MIN_EYE_BOX_PADDING);

    let x = (min_x - horizontal_padding).max(0.0).round() as i32;
    let y = (min_y - vertical_padding).max(0.0).round() as i32;
    let width = (eye_width + (horizontal_padding * 2.0)).round() as i32;
    let height = (eye_height + (vertical_padding * 2.0)).round() as i32;

    Some(Rect::new(x, y, width.max(1), height.max(1)))
}

fn draw_eye_bounding_box(frame: &mut Mat, eye: Rect) -> opencv::Result<()> {
    // azul para las cajas de los ojos
    imgproc::rectangle(
        frame,
        eye,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    Ok(())
}

fn draw_closed_eye_fill(frame: &mut Mat, eye: Rect) -> opencv::Result<()> {
    imgproc::rectangle(
        frame,
        eye,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    Ok(())
}

fn build_eye_analysis_box_from_corners(
    facial_points: &Vector<Point2f>,
    eye_indices: &[usize; 6],
) -> Rect {
    let left_corner = facial_points
        .get(eye_indices[0])
        .unwrap_or(Point2f::new(0.0, 0.0));
    let right_corner = facial_points
        .get(eye_indices[3])
        .unwrap_or(Point2f::new(0.0, 0.0));

    let center_x = ((left_corner.x + right_corner.x) * 0.5).round();
    let center_y = ((left_corner.y + right_corner.y) * 0.5).round();
    let eye_width = euclidean_distance(left_corner, right_corner).max(1.0);
    let analysis_width = (eye_width * EYE_ANALYSIS_WIDTH_FACTOR).round() as i32;
    let analysis_height = (eye_width * EYE_ANALYSIS_HEIGHT_FACTOR).round() as i32;

    Rect::new(
        (center_x as i32) - (analysis_width / 2),
        (center_y as i32) - (analysis_height / 2),
        analysis_width.max(1),
        analysis_height.max(1),
    )
}

fn is_eye_closed(
    eye_aspect_ratio: f32,
    visual_metrics: EyeVisualMetrics,
) -> bool {
    let is_closed_by_ear = eye_aspect_ratio <= EYE_CLOSED_EAR_THRESHOLD;
    let is_closed_by_visual = visual_metrics.span_ratio <= VISUAL_CLOSED_SPAN_RATIO_THRESHOLD
        && visual_metrics.dark_ratio <= VISUAL_CLOSED_DARK_RATIO_THRESHOLD;
    let has_eyelid_line = visual_metrics.eyelid_line_ratio >= EYELID_LINE_MIN_LENGTH_RATIO;
    let is_closed_by_hybrid = eye_aspect_ratio <= HYBRID_EAR_SOFT_THRESHOLD
        && (is_closed_by_visual || has_eyelid_line);

    is_closed_by_ear || is_closed_by_hybrid
}

fn get_eye_status_label(
    is_closed: bool,
    eye_name: &str,
) -> String {
    if is_closed {
        format!("{eye_name} cerrado")
    } else {
        format!("{eye_name} abierto")
    }
}

fn compute_eye_aspect_ratio(
    facial_points: &Vector<Point2f>,
    eye_indices: &[usize; 6],
) -> f32 {
    let eye_points: Vec<Point2f> = eye_indices
        .iter()
        .filter_map(|&index| facial_points.get(index).ok())
        .collect();

    if eye_points.len() != 6 {
        return 1.0;
    }

    let vertical_distance_1 = euclidean_distance(eye_points[1], eye_points[5]);
    let vertical_distance_2 = euclidean_distance(eye_points[2], eye_points[4]);
    let horizontal_distance = euclidean_distance(eye_points[0], eye_points[3]);

    if horizontal_distance <= f32::EPSILON {
        return 1.0;
    }

    (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
}

fn euclidean_distance(point_a: Point2f, point_b: Point2f) -> f32 {
    let delta_x = point_a.x - point_b.x;
    let delta_y = point_a.y - point_b.y;
    (delta_x * delta_x + delta_y * delta_y).sqrt()
}

#[derive(Clone, Copy)]
struct EyeVisualMetrics {
    span_ratio: f32,
    dark_ratio: f32,
    eyelid_line_ratio: f32,
}

fn compute_eye_visual_metrics(gray_frame: &Mat, eye_box: Rect) -> opencv::Result<EyeVisualMetrics> {
    let clipped_eye_box = clamp_rect_to_frame(eye_box, gray_frame.size()?);
    if clipped_eye_box.width <= 1 || clipped_eye_box.height <= 1 {
        return Ok(EyeVisualMetrics {
            span_ratio: 1.0,
            dark_ratio: 1.0,
            eyelid_line_ratio: 0.0,
        });
    }

    let focus_box = build_visual_focus_box(clipped_eye_box);
    let eye_roi = gray_frame.roi(focus_box)?;
    let mut blurred_roi = Mat::default();
    imgproc::gaussian_blur(
        &eye_roi,
        &mut blurred_roi,
        Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
    )?;

    let mut mean = Mat::default();
    let mut stddev = Mat::default();
    core::mean_std_dev(&blurred_roi, &mut mean, &mut stddev, &Mat::default())?;
    let mean_intensity = *mean.at_2d::<f64>(0, 0)?;
    let stddev_intensity = *stddev.at_2d::<f64>(0, 0)?;
    let adaptive_threshold = (mean_intensity - (stddev_intensity * 0.35)).clamp(25.0, 105.0);

    let mut dark_mask = Mat::default();
    imgproc::threshold(
        &blurred_roi,
        &mut dark_mask,
        adaptive_threshold,
        255.0,
        imgproc::THRESH_BINARY_INV,
    )?;

    let row_activation_threshold = (f64::from(focus_box.width) * VISUAL_DARK_ROW_MIN_RATIO)
        .ceil()
        .max(1.0) as i32;
    let mut first_active_row = None;
    let mut last_active_row = None;

    for row in 0..focus_box.height {
        let row_roi = dark_mask.roi(Rect::new(0, row, focus_box.width, 1))?;
        let active_pixels = core::count_non_zero(&row_roi)?;

        if active_pixels >= row_activation_threshold {
            if first_active_row.is_none() {
                first_active_row = Some(row);
            }
            last_active_row = Some(row);
        }
    }

    let span_ratio = if let (Some(first_row), Some(last_row)) = (first_active_row, last_active_row)
    {
        let active_span = (last_row - first_row + 1) as f32;
        active_span / focus_box.height as f32
    } else {
        1.0
    };

    let active_pixels = core::count_non_zero(&dark_mask)? as f32;
    let total_pixels = (focus_box.width * focus_box.height).max(1) as f32;
    let dark_ratio = active_pixels / total_pixels;
    let eyelid_line_ratio = compute_eyelid_line_ratio(&dark_mask, focus_box.width, focus_box.height)?;

    Ok(EyeVisualMetrics {
        span_ratio,
        dark_ratio,
        eyelid_line_ratio,
    })
}

fn compute_eyelid_line_ratio(mask: &Mat, width: i32, height: i32) -> opencv::Result<f32> {
    let mut edges = Mat::default();
    imgproc::canny(mask, &mut edges, 50.0, 150.0, 3, false)?;

    let mut lines = Vector::<Vec4i>::new();
    imgproc::hough_lines_p(
        &edges,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        8,
        f64::from(width) * 0.30,
        3.0,
    )?;

    let mut best_ratio = 0.0_f32;

    for line in lines.iter() {
        let dx = (line[2] - line[0]) as f32;
        let dy = (line[3] - line[1]) as f32;
        let length = (dx * dx + dy * dy).sqrt();
        if length <= f32::EPSILON {
            continue;
        }

        let vertical_drift_ratio = dy.abs() / (height.max(1) as f32);
        if vertical_drift_ratio > EYELID_LINE_MAX_VERTICAL_DRIFT {
            continue;
        }

        best_ratio = best_ratio.max(length / width.max(1) as f32);
    }

    Ok(best_ratio)
}

fn clamp_rect_to_frame(rect: Rect, frame_size: Size) -> Rect {
    let x = rect.x.clamp(0, frame_size.width.saturating_sub(1));
    let y = rect.y.clamp(0, frame_size.height.saturating_sub(1));
    let max_width = (frame_size.width - x).max(0);
    let max_height = (frame_size.height - y).max(0);
    let width = rect.width.min(max_width).max(0);
    let height = rect.height.min(max_height).max(0);

    Rect::new(x, y, width, height)
}

fn build_visual_focus_box(eye_box: Rect) -> Rect {
    let focus_width = ((eye_box.width as f32) * VISUAL_FOCUS_WIDTH_FACTOR).round() as i32;
    let focus_height = ((eye_box.height as f32) * VISUAL_FOCUS_HEIGHT_FACTOR).round() as i32;
    let width = focus_width.clamp(1, eye_box.width);
    let height = focus_height.clamp(1, eye_box.height);
    let x = eye_box.x + ((eye_box.width - width) / 2);
    let y = eye_box.y + ((eye_box.height - height) / 2);

    Rect::new(x, y, width, height)
}

fn draw_eye_status_label(
    frame: &mut Mat,
    eye: Rect,
    label: &str,
    ear: f32,
    visual_metrics: EyeVisualMetrics,
) -> opencv::Result<()> {
    let frame_height = frame.size()?.height;
    let preferred_text_origin_y = eye.y - 52;
    let text_origin_y = if preferred_text_origin_y >= 18 {
        preferred_text_origin_y
    } else {
        (eye.y + eye.height + 20).min(frame_height.saturating_sub(40))
    };
    let text_origin = Point::new(eye.x, text_origin_y);
    let ear_text = format!("EAR: {:.3}", ear);
    let visual_text = format!(
        "VIS: {:.3} DARK: {:.3} LINE: {:.3}",
        visual_metrics.span_ratio, visual_metrics.dark_ratio, visual_metrics.eyelid_line_ratio
    );
    let ear_origin = Point::new(text_origin.x, text_origin.y + 18);
    let visual_origin = Point::new(text_origin.x, text_origin.y + 36);

    imgproc::put_text(
        frame,
        label,
        text_origin,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.55,
        Scalar::new(0.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &ear_text,
        ear_origin,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(255.0, 255.0, 0.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &visual_text,
        visual_origin,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(0.0, 200.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

fn draw_visual_legend(frame: &mut Mat) -> opencv::Result<()> {
    let legend_box = Rect::new(20, 20, 320, 125);

    imgproc::rectangle(
        frame,
        legend_box,
        Scalar::new(40.0, 40.0, 40.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    imgproc::rectangle(
        frame,
        legend_box,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    draw_legend_item(frame, 35, Scalar::new(0.0, 255.0, 0.0, 0.0), "Rostro detectado")?;
    draw_legend_item(frame, 65, Scalar::new(255.0, 0.0, 0.0, 0.0), "Caja del ojo")?;
    draw_legend_item(frame, 95, Scalar::new(0.0, 0.0, 255.0, 0.0), "Landmarks faciales")?;
    draw_legend_text(
        frame,
        "Texto amarillo: ojo abierto/cerrado",
        Point::new(35, 128),
    )?;

    Ok(())
}

fn draw_legend_item(
    frame: &mut Mat,
    y: i32,
    color: Scalar,
    label: &str,
) -> opencv::Result<()> {
    imgproc::rectangle(
        frame,
        Rect::new(35, y, 20, 12),
        color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    draw_legend_text(frame, label, Point::new(70, y + 11))?;

    Ok(())
}

fn draw_legend_text(frame: &mut Mat, text: &str, origin: Point) -> opencv::Result<()> {
    imgproc::put_text(
        frame,
        text,
        origin,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}
