mod analysis;
mod config;
mod io;
mod overlay;
mod processing;

use std::collections::VecDeque;

use opencv::{highgui, prelude::*, videoio};

use crate::config::{
    FACEMARK_MODEL_PATH, FACE_CASCADE_PATH, INPUT_VIDEO_PATH, ORIGINAL_WINDOW_NAME,
    OUTPUT_VIDEO_PATH, PROCESSED_WINDOW_NAME,
};
use crate::io::{
    calculate_frame_delay, create_output_writer, ensure_output_directory_exists, load_cascade_classifier,
    load_facemark_model, open_input_video,
};
use crate::overlay::draw_ear_graph;
use crate::processing::{process_frame, update_ear_history};

fn main() -> opencv::Result<()> {
    ensure_output_directory_exists()?;

    let mut capture = open_input_video(INPUT_VIDEO_PATH)?;
    let fps = capture.get(videoio::CAP_PROP_FPS)?;
    let frame_width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let mut writer = create_output_writer(OUTPUT_VIDEO_PATH, fps, frame_width, frame_height)?;

    let mut face_detector = load_cascade_classifier(FACE_CASCADE_PATH)?;
    let mut facemark = load_facemark_model(FACEMARK_MODEL_PATH)?;

    let frame_delay = calculate_frame_delay(fps);

    highgui::named_window(ORIGINAL_WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(ORIGINAL_WINDOW_NAME, 960, 540)?;
    highgui::named_window(PROCESSED_WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(PROCESSED_WINDOW_NAME, 960, 540)?;

    println!("Video de entrada: {}", INPUT_VIDEO_PATH);
    println!("Resolucion detectada: {}x{}", frame_width, frame_height);
    println!("FPS detectados: {:.2}", fps);
    println!("Video de salida: {}", OUTPUT_VIDEO_PATH);
    println!("Presiona la tecla 'q' para cerrar la ventana.");

    let mut processed_frames = 0;
    let mut ear_history = VecDeque::<f32>::new();

    loop {
        let mut frame = opencv::core::Mat::default();
        let was_frame_read = capture.read(&mut frame)?;

        if !was_frame_read || frame.empty() {
            break;
        }

        let mut processing_result = process_frame(&frame, &mut face_detector, &mut facemark)?;
        update_ear_history(&mut ear_history, processing_result.average_ear);
        draw_ear_graph(&mut processing_result.frame, &ear_history)?;

        highgui::imshow(ORIGINAL_WINDOW_NAME, &frame)?;
        highgui::imshow(PROCESSED_WINDOW_NAME, &processing_result.frame)?;

        writer.write(&processing_result.frame)?;

        processed_frames += 1;

        let pressed_key = highgui::wait_key(frame_delay)?;
        if pressed_key == i32::from(b'q') {
            break;
        }
    }

    println!("Frames procesados: {}", processed_frames);
    println!("Procesamiento finalizado.");

    Ok(())
}
