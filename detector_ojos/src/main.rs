use std::fs; //crear carpetas y manejar archivos
use std::path::Path;

use opencv::{ 
    core::{Mat, Point, Point2f, Rect, Scalar, Size, Vector}, //Mat para imágenes, Point para el centro, Point2f para landmarks, Rect para bounding boxes, Scalar para colores, Size para dimensiones
    face, //face para usar FacemarkLBF y dibujar landmarks
    highgui, //highgui para mostrar ventanas y manejar eventos de teclado
    imgproc, //imgproc para convertir a gris y dibujar rectángulos
    objdetect, //objdetect para usar los clasificadores Haar Cascade
    prelude::*, //prelude para traer a scope varias funciones y tipos comunes de OpenCV
    videoio, //videoio para leer y escribir videos, manejar cámaras 
};

// indica dónde se espera encontrar el video de entrada
const INPUT_VIDEO_PATH: &str = "videos/input_video.mp4";

// define la carpeta donde guardaremos los videos procesados
const OUTPUT_DIRECTORY: &str = "videos_salida";

// nombre del archivo de salida
const OUTPUT_VIDEO_PATH: &str = "videos_salida/video_procesado.mp4";

// nombre de la ventana donde se mostrará el video mientras se procesa
const WINDOW_NAME: &str = "Vista previa del video";

// ruta del clasificador que detecta rostros frontales
const FACE_CASCADE_PATH: &str = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

// ruta del modelo preentrenado de FacemarkLBF
const FACEMARK_MODEL_PATH: &str = "models/lbfmodel.yaml";

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

    // Creamos la ventana donde se mostrará el video mientras se procesa
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(WINDOW_NAME, 960, 540)?;

    // se muestra por consola los datos principales del video
    println!("Video de entrada: {}", INPUT_VIDEO_PATH);
    println!("Resolucion detectada: {}x{}", frame_width, frame_height);
    println!("FPS detectados: {:.2}", fps);
    println!("Video de salida: {}", OUTPUT_VIDEO_PATH);
    println!("Presiona la tecla 'q' para cerrar la ventana.");

    // contador que ayuda a saber cuántos frames se han procesado
    let mut processed_frames = 0;

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
        let processed_frame = process_frame(&frame, &mut face_detector, &mut facemark)?;

        // muestra el frame procesado en la ventana creada al inicio del programa
        highgui::imshow(WINDOW_NAME, &processed_frame)?;

        // guarda el frame dentro del vídeo de salida
        writer.write(&processed_frame)?;

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
) -> opencv::Result<Mat> { //función que procesa cada frame
    // crea un frame de salida para dibujar sobre él sin alterar el frame original
    let mut output_frame = Mat::default();
    frame.copy_to(&mut output_frame)?;

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
        }
    }

    Ok(output_frame)
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
