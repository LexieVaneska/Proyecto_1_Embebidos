use std::collections::VecDeque; //historial de EAR

use opencv::{
    core::{Mat, Point2f, Rect, Vector}, //tipos de opencv
    face, imgproc, objdetect, //módulos de openCV
    prelude::*,
};

use crate::analysis::{ //funciones de análisis
    build_eye_analysis_box_from_corners, build_eye_box_from_indices, compute_eye_aspect_ratio,
    compute_eye_visual_metrics, get_eye_status_label, is_eye_closed, select_primary_face,
};
use crate::config::{EAR_HISTORY_MAX_POINTS, LEFT_EYE_INDICES, RIGHT_EYE_INDICES}; //constantes de configuración
use crate::overlay::{ //funciones de overlay
    draw_closed_eye_fill, draw_eye_bounding_box, draw_eye_status_label, draw_face_bounding_box,
    draw_landmarks, draw_visual_legend,
};

pub struct FrameProcessingResult { //lo que devuelve el procesamiento de cada frame
    pub frame: Mat, //el frame ya procesado con los overlays
    pub average_ear: Option<f32>,
}

pub fn process_frame( //función principal para procesar cada frame
    frame: &Mat,
    face_detector: &mut objdetect::CascadeClassifier,
    facemark: &mut opencv::core::Ptr<face::FacemarkLBF>,
) -> opencv::Result<FrameProcessingResult> { //recibe el frame original, el detector de rostros y el modelo de landmarks, devuelve el frame procesado y el EAR promedio
    let mut output_frame = Mat::default(); //crea un output frame vacío
    frame.copy_to(&mut output_frame)?;//copia el frame original
    let mut average_ear = None; //inicializa el EAR promedio como None
    // convierte el frame de color a escala de grises para mejorar la detección de rostros
    let mut gray_frame = Mat::default();
    imgproc::cvt_color(frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;
    //ecualiza el histograma para mejorar el contraste, lo que ayuda a detectar rostros en condiciones de iluminación difíciles
    let mut equalized_frame = Mat::default();
    imgproc::equalize_hist(&gray_frame, &mut equalized_frame)?;
    //detecta rostros en el frame ecualizado
    let mut faces = Vector::<Rect>::new();
    face_detector.detect_multi_scale(
        &equalized_frame,
        &mut faces,
        1.1,
        5,
        objdetect::CASCADE_SCALE_IMAGE,
        opencv::core::Size::new(80, 80), //evita detectar rostros demasiado pequeños
        opencv::core::Size::new(0, 0),
    )?;
    //rostro principal seleccionado para análisis
    if let Some(face) = select_primary_face(&faces, frame.size()?) {
        draw_face_bounding_box(&mut output_frame, face)?; //dibuja un recuadro alrededor del rostro seleccionado
        //prepara el rostro para landmarks: crea un vector con el rostro seleccionado, necesario para la función fit del modelo de landmarks
        let mut selected_faces = Vector::<Rect>::new(); 
        selected_faces.push(face);
        //detección de landmarks faciales en el rostro seleccionado
        let mut landmarks = Vector::<Vector<Point2f>>::new();
        let landmarks_found = facemark.fit(frame, &selected_faces, &mut landmarks)?;
        //validación de landmarks
        if landmarks_found && !landmarks.is_empty() {
            draw_landmarks(&mut output_frame, &landmarks)?; //dibuja los landmarks detectados en el rostro
            average_ear = draw_eye_boxes_and_labels_from_landmarks( //llamada al análisis de ojso
                &mut output_frame,
                &equalized_frame,
                &landmarks,
            )?;
        }
    }

    draw_visual_legend(&mut output_frame)?; //leyenda visual para explicar los indicadores en pantalla

    Ok(FrameProcessingResult { //retorno de resultados del procesamiento del frame
        frame: output_frame,
        average_ear,
    })
}

pub fn update_ear_history(ear_history: &mut VecDeque<f32>, average_ear: Option<f32>) { //mantiene una ventana de los últimos valores de EAR para mostrar una gráfica de tendencia
    if let Some(ear) = average_ear {
        ear_history.push_back(ear);
        while ear_history.len() > EAR_HISTORY_MAX_POINTS {
            ear_history.pop_front();
        }
    }
}

fn draw_eye_boxes_and_labels_from_landmarks( //función para analizar los ojos a partir de los landmarks detectados
    frame: &mut Mat,
    gray_frame: &Mat,
    landmarks: &Vector<Vector<Point2f>>,
) -> opencv::Result<Option<f32>> {
    let mut detected_ears = Vec::<f32>::new(); //inicialización del vector de EAR detectados en el frame, se usará para calcular el promedio al final

    for facial_points in landmarks.iter() { //recorrido de landmarks detectados
        if facial_points.len() < 48 { //validación mínima de puntos faciales para asegurar que se detectaron los landmarks necesarios para analizar los ojos
            continue;
        }

        let mut left_eye_detection = None; //variables temporales para cada ojo
        let mut right_eye_detection = None;
        //bloque del ojo izquierdo
        if let Some(left_eye_box) = build_eye_box_from_indices(&facial_points, &LEFT_EYE_INDICES) {
            let left_eye_ear = compute_eye_aspect_ratio(&facial_points, &LEFT_EYE_INDICES);
            detected_ears.push(left_eye_ear);
            let left_eye_analysis_box =
                build_eye_analysis_box_from_corners(&facial_points, &LEFT_EYE_INDICES);
            let left_eye_visual_metrics =
                compute_eye_visual_metrics(gray_frame, left_eye_analysis_box)?;
            let left_eye_is_closed = is_eye_closed(left_eye_ear, left_eye_visual_metrics);
            let left_eye_status = get_eye_status_label(left_eye_is_closed, "Ojo izquierdo");
            //guarda toda la información del ojo izquierdo en una tupla para usarla después al dibujar los overlays
            left_eye_detection = Some((
                left_eye_box,
                left_eye_status,
                left_eye_ear,
                left_eye_visual_metrics,
                left_eye_is_closed,
            ));
        }
        //bloque del ojo derecho, mismo proceso que el izquierdo
        if let Some(right_eye_box) = build_eye_box_from_indices(&facial_points, &RIGHT_EYE_INDICES) {
            let right_eye_ear = compute_eye_aspect_ratio(&facial_points, &RIGHT_EYE_INDICES);
            detected_ears.push(right_eye_ear);
            let right_eye_analysis_box =
                build_eye_analysis_box_from_corners(&facial_points, &RIGHT_EYE_INDICES);
            let right_eye_visual_metrics =
                compute_eye_visual_metrics(gray_frame, right_eye_analysis_box)?;
            let right_eye_is_closed = is_eye_closed(right_eye_ear, right_eye_visual_metrics);
            let right_eye_status = get_eye_status_label(right_eye_is_closed, "Ojo derecho");
            right_eye_detection = Some((
                right_eye_box,
                right_eye_status,
                right_eye_ear,
                right_eye_visual_metrics,
                right_eye_is_closed,
            ));
        }
        //si ambos ojos están detectados y ambos están cerrados, dibuja un relleno rojo para indicar que ambos ojos están cerrados
        if let (Some(left_eye), Some(right_eye)) = (&left_eye_detection, &right_eye_detection) {
            if left_eye.4 && right_eye.4 {
                draw_closed_eye_fill(frame, left_eye.0)?;
                draw_closed_eye_fill(frame, right_eye.0)?;
            }
        }
        //dibuja la caja y la información de estado para el ojo izquierdo 
        if let Some((left_eye_box, left_eye_status, left_eye_ear, left_eye_visual_metrics, _)) =
            left_eye_detection
        {
            //dibuja el recuadro del ojo izquierdo
            draw_eye_bounding_box(frame, left_eye_box)?;
            draw_eye_status_label( //escribe el estado y las metricas del ojo izquierdo
                frame,
                left_eye_box,
                &left_eye_status,
                left_eye_ear,
                left_eye_visual_metrics,
            )?;
        }
        //dibuja la caja y la información de estado para el ojo derecho
        if let Some((
            right_eye_box,
            right_eye_status,
            right_eye_ear,
            right_eye_visual_metrics,
            _,
        )) = right_eye_detection
        {
            draw_eye_bounding_box( //traza el recuadro del ojo derecho
                frame,
                right_eye_box,
            )?;
            draw_eye_status_label( //escribe el estado y las metricas del ojo derecho
                frame,
                right_eye_box,
                &right_eye_status,
                right_eye_ear,
                right_eye_visual_metrics,
            )?;
        }
    }
    //si no se detectaron ojos, devuelve None, de lo contrario calcula el promedio de EAR detectados y lo devuelve para mostrarlo en la gráfica de tendencia
    if detected_ears.is_empty() {
        Ok(None)
    } else {
        let ear_sum: f32 = detected_ears.iter().copied().sum(); //suma de los EAR detectados
        Ok(Some(ear_sum / detected_ears.len() as f32)) //|promedio de EAR detectados 
    }
}
