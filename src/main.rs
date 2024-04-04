extern crate opencv;
use opencv::core;
use opencv::features2d;
use opencv::imgcodecs;
use opencv::prelude::*;
use opencv::core::Vector;

#[derive(Debug,Clone,Copy)]
struct Keypoint {
    x: f32,
    y: f32,
    angle: f32,
}

fn extract_feature_points<Keypoint>(img: &core::Mat) -> opencv::Result<(Vec<Keypoint>, core::Mat)> {
    let descriptor_type = features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB;
    let descriptor_size = 0; // MLDBのデフォルトサイズ
    let descriptor_channels = 3; // MLDBデフォルトチャンネル数
    let threshold = 0.001f32; // 検出器のデフォルト閾値
    let n_octaves = 4; // デフォルトのオクターブ数
    let n_octave_layers = 4; // オクターブあたりのデフォルトレイヤー数
    let diffusivity = opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2;

    let mut detector = opencv::features2d::AKAZE::create(
        descriptor_type,
        descriptor_size,
        descriptor_channels,
        threshold,
        n_octaves,
        n_octave_layers,
        diffusivity,
    )?;

    let mut keypoints: Vector<core::KeyPoint> = Vector::new();
    let mut descriptors = core::Mat::default();
    detector.detect_and_compute(
        img,
        &core::no_array(),
        &mut keypoints,
        &mut descriptors,
        false,
    )?;

    let keypoints: Vec<crate::Keypoint> = keypoints
        .iter()
        .map(|kp| {
            crate::Keypoint {
                x: kp.pt().x,
                y: kp.pt().y,
                angle: kp.angle(),
            }
        })
        .collect();

    Ok((keypoints, descriptors))
}

fn main() {
    let img = imgcodecs::imread("path/to/image.jpg", imgcodecs::IMREAD_GRAYSCALE).unwrap();
    extract_feature_points::<core::KeyPoint>(&img).unwrap();
}
