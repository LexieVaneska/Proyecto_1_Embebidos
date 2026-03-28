use std::fs; //crear carpetas y manejar archivos
use std::path::Path;

use opencv::{ 
    core::{Mat, Size}, //Mat para estructura de datos para imágenes, Size para dimensiones
    highgui, //highgui para mostrar ventanas y manejar eventos de teclado
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

        
        
        // etapa de procesamiento POR EL MOMENTO solo copia el frame 
        // AQUÍ irá rostro, ojos y máscara
        let processed_frame = process_frame(&frame)?;

        // muestra el frame procesado en la ventana creada al inicio del programa
        highgui::imshow(WINDOW_NAME, &processed_frame)?;

        // guarda el frame dentro del vídeo de salida
        writer.write(&processed_frame)?;

        processed_frames += 1; //suma uno al contador

        // si se presiona q, el programa termina
        let pressed_key = highgui::wait_key(1)?;
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

fn process_frame(frame: &Mat) -> opencv::Result<Mat> { //función que procesa cada frame
    //crea un frame de salida vacío, copia el contenido del original y lo devuelve
    let mut output_frame = Mat::default();
    frame.copy_to(&mut output_frame)?;
    Ok(output_frame)
}
