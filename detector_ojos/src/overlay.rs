use std::collections::VecDeque;

use opencv::{
    core::{Mat, Point, Rect, Scalar},
    face,
    imgproc,
    prelude::*,
};

use crate::analysis::EyeVisualMetrics;
use crate::config::{
    EAR_GRAPH_HEIGHT, EAR_GRAPH_MARGIN, EAR_GRAPH_MAX_VALUE, EAR_GRAPH_MIN_VALUE, EAR_GRAPH_WIDTH,
    EAR_HISTORY_MAX_POINTS, EYE_CLOSED_EAR_THRESHOLD,
};

pub fn draw_face_bounding_box(frame: &mut Mat, face: Rect) -> opencv::Result<()> {
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

pub fn draw_landmarks(
    frame: &mut Mat,
    landmarks: &opencv::core::Vector<opencv::core::Vector<opencv::core::Point2f>>,
) -> opencv::Result<()> {
    for facial_points in landmarks.iter() {
        face::draw_facemarks(frame, &facial_points, Scalar::new(0.0, 0.0, 255.0, 0.0))?;
    }
    Ok(())
}

pub fn draw_eye_bounding_box(frame: &mut Mat, eye: Rect) -> opencv::Result<()> {
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

pub fn draw_closed_eye_fill(frame: &mut Mat, eye: Rect) -> opencv::Result<()> {
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

pub fn draw_eye_status_label(
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

pub fn draw_visual_legend(frame: &mut Mat) -> opencv::Result<()> {
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

pub fn draw_ear_graph(frame: &mut Mat, ear_history: &VecDeque<f32>) -> opencv::Result<()> {
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

fn draw_legend_item(frame: &mut Mat, y: i32, color: Scalar, label: &str) -> opencv::Result<()> {
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
