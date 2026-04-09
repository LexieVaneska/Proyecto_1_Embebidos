use opencv::{
    core::{self, Point, Point2f, Rect, Size, Vec4i, Vector},
    imgproc,
    prelude::*,
};

use crate::config::{
    EYE_ANALYSIS_HEIGHT_FACTOR, EYE_ANALYSIS_WIDTH_FACTOR, EYE_BOX_HORIZONTAL_PADDING_FACTOR,
    EYE_BOX_VERTICAL_PADDING_FACTOR, EYE_CLOSED_EAR_THRESHOLD, EYELID_LINE_MAX_VERTICAL_DRIFT,
    EYELID_LINE_MIN_LENGTH_RATIO, HYBRID_EAR_SOFT_THRESHOLD, MIN_EYE_BOX_PADDING,
    VISUAL_CLOSED_DARK_RATIO_THRESHOLD, VISUAL_CLOSED_SPAN_RATIO_THRESHOLD,
    VISUAL_DARK_ROW_MIN_RATIO, VISUAL_FOCUS_HEIGHT_FACTOR, VISUAL_FOCUS_WIDTH_FACTOR,
};

#[derive(Clone, Copy)]
pub struct EyeVisualMetrics {
    pub span_ratio: f32,
    pub dark_ratio: f32,
    pub eyelid_line_ratio: f32,
}

pub fn select_primary_face(faces: &Vector<Rect>, frame_size: Size) -> Option<Rect> {
    if faces.is_empty() {
        return None;
    }

    let frame_center = Point::new(frame_size.width / 2, frame_size.height / 2);
    let mut best_face = None;
    let mut best_score = f64::MIN;

    for face in faces.iter() {
        let face_area = f64::from(face.width * face.height);
        let face_center = Point::new(face.x + face.width / 2, face.y + face.height / 2);

        let distance_x = f64::from(face_center.x - frame_center.x);
        let distance_y = f64::from(face_center.y - frame_center.y);
        let distance_to_center = (distance_x * distance_x + distance_y * distance_y).sqrt();

        let score = face_area - (distance_to_center * 250.0);

        if score > best_score {
            best_score = score;
            best_face = Some(face);
        }
    }

    best_face
}

pub fn build_eye_box_from_indices(
    facial_points: &Vector<Point2f>,
    eye_indices: &[usize],
) -> Option<Rect> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for &index in eye_indices {
        let point = facial_points.get(index).ok()?;
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }

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

pub fn build_eye_analysis_box_from_corners(
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

pub fn is_eye_closed(eye_aspect_ratio: f32, visual_metrics: EyeVisualMetrics) -> bool {
    let is_closed_by_ear = eye_aspect_ratio <= EYE_CLOSED_EAR_THRESHOLD;
    let is_closed_by_visual = visual_metrics.span_ratio <= VISUAL_CLOSED_SPAN_RATIO_THRESHOLD
        && visual_metrics.dark_ratio <= VISUAL_CLOSED_DARK_RATIO_THRESHOLD;
    let has_eyelid_line = visual_metrics.eyelid_line_ratio >= EYELID_LINE_MIN_LENGTH_RATIO;
    let is_closed_by_hybrid =
        eye_aspect_ratio <= HYBRID_EAR_SOFT_THRESHOLD && (is_closed_by_visual || has_eyelid_line);

    is_closed_by_ear || is_closed_by_hybrid
}

pub fn get_eye_status_label(is_closed: bool, eye_name: &str) -> String {
    if is_closed {
        format!("{eye_name} cerrado")
    } else {
        format!("{eye_name} abierto")
    }
}

pub fn compute_eye_aspect_ratio(
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

pub fn compute_eye_visual_metrics(gray_frame: &Mat, eye_box: Rect) -> opencv::Result<EyeVisualMetrics> {
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
    let eyelid_line_ratio =
        compute_eyelid_line_ratio(&dark_mask, focus_box.width, focus_box.height)?;

    Ok(EyeVisualMetrics {
        span_ratio,
        dark_ratio,
        eyelid_line_ratio,
    })
}

fn euclidean_distance(point_a: Point2f, point_b: Point2f) -> f32 {
    let delta_x = point_a.x - point_b.x;
    let delta_y = point_a.y - point_b.y;
    (delta_x * delta_x + delta_y * delta_y).sqrt()
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
