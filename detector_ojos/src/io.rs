use std::fs;
use std::path::Path;

use opencv::{
    face,
    prelude::*,
    videoio,
    core::Size,
};

use crate::config::OUTPUT_DIRECTORY;

pub fn ensure_output_directory_exists() -> opencv::Result<()> {
    if Path::new(OUTPUT_DIRECTORY).exists() {
        return Ok(());
    }

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

pub fn open_input_video(video_path: &str) -> opencv::Result<videoio::VideoCapture> {
    let capture = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;

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

pub fn load_cascade_classifier(
    cascade_path: &str,
) -> opencv::Result<opencv::objdetect::CascadeClassifier> {
    if !Path::new(cascade_path).exists() {
        return Err(opencv::Error::new(
            0,
            format!("No se encontró el clasificador Haar Cascade en '{}'", cascade_path),
        ));
    }

    let classifier = opencv::objdetect::CascadeClassifier::new(cascade_path)?;
    Ok(classifier)
}

pub fn load_facemark_model(
    model_path: &str,
) -> opencv::Result<opencv::core::Ptr<face::FacemarkLBF>> {
    if !Path::new(model_path).exists() {
        return Err(opencv::Error::new(
            0,
            format!("No se encontró el modelo de Facemark en '{}'", model_path),
        ));
    }

    let mut facemark = face::FacemarkLBF::create_def()?;
    facemark.load_model(model_path)?;

    Ok(facemark)
}

pub fn calculate_frame_delay(fps: f64) -> i32 {
    if fps > 0.0 {
        (1000.0 / fps).round() as i32
    } else {
        33
    }
}

pub fn create_output_writer(
    output_path: &str,
    fps: f64,
    frame_width: i32,
    frame_height: i32,
) -> opencv::Result<videoio::VideoWriter> {
    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let output_fps = if fps > 0.0 { fps } else { 30.0 };

    let writer = videoio::VideoWriter::new(
        output_path,
        fourcc,
        output_fps,
        Size::new(frame_width, frame_height),
        true,
    )?;

    Ok(writer)
}
