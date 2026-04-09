use std::collections::VecDeque;

use opencv::{
    core::{Mat, Point2f, Rect, Vector},
    face, imgproc, objdetect,
    prelude::*,
};

use crate::analysis::{
    build_eye_analysis_box_from_corners, build_eye_box_from_indices, compute_eye_aspect_ratio,
    compute_eye_visual_metrics, get_eye_status_label, is_eye_closed, select_primary_face,
};
use crate::config::{EAR_HISTORY_MAX_POINTS, LEFT_EYE_INDICES, RIGHT_EYE_INDICES};
use crate::overlay::{
    draw_closed_eye_fill, draw_eye_bounding_box, draw_eye_status_label, draw_face_bounding_box,
    draw_landmarks, draw_visual_legend,
};

pub struct FrameProcessingResult {
    pub frame: Mat,
    pub average_ear: Option<f32>,
}

pub fn process_frame(
    frame: &Mat,
    face_detector: &mut objdetect::CascadeClassifier,
    facemark: &mut opencv::core::Ptr<face::FacemarkLBF>,
) -> opencv::Result<FrameProcessingResult> {
    let mut output_frame = Mat::default();
    frame.copy_to(&mut output_frame)?;
    let mut average_ear = None;

    let mut gray_frame = Mat::default();
    imgproc::cvt_color(frame, &mut gray_frame, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut equalized_frame = Mat::default();
    imgproc::equalize_hist(&gray_frame, &mut equalized_frame)?;

    let mut faces = Vector::<Rect>::new();
    face_detector.detect_multi_scale(
        &equalized_frame,
        &mut faces,
        1.1,
        5,
        objdetect::CASCADE_SCALE_IMAGE,
        opencv::core::Size::new(80, 80),
        opencv::core::Size::new(0, 0),
    )?;

    if let Some(face) = select_primary_face(&faces, frame.size()?) {
        draw_face_bounding_box(&mut output_frame, face)?;

        let mut selected_faces = Vector::<Rect>::new();
        selected_faces.push(face);

        let mut landmarks = Vector::<Vector<Point2f>>::new();
        let landmarks_found = facemark.fit(frame, &selected_faces, &mut landmarks)?;

        if landmarks_found && !landmarks.is_empty() {
            draw_landmarks(&mut output_frame, &landmarks)?;
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

pub fn update_ear_history(ear_history: &mut VecDeque<f32>, average_ear: Option<f32>) {
    if let Some(ear) = average_ear {
        ear_history.push_back(ear);
        while ear_history.len() > EAR_HISTORY_MAX_POINTS {
            ear_history.pop_front();
        }
    }
}

fn draw_eye_boxes_and_labels_from_landmarks(
    frame: &mut Mat,
    gray_frame: &Mat,
    landmarks: &Vector<Vector<Point2f>>,
) -> opencv::Result<Option<f32>> {
    let mut detected_ears = Vec::<f32>::new();

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
            let left_eye_status = get_eye_status_label(left_eye_is_closed, "Ojo izquierdo");
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
            let right_eye_status = get_eye_status_label(right_eye_is_closed, "Ojo derecho");
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
            draw_eye_bounding_box(
                frame,
                right_eye_box,
            )?;
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
