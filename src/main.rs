extern crate opencv;
use opencv::core;
use opencv::core::Vector;
use opencv::features2d;
use opencv::imgcodecs;
use opencv::prelude::*;
use serde;
use serde_json;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Keypoint {
    x: f32,
    y: f32,
    angle: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KeypointList {
    keypoints: Vec<crate::Keypoint>,
}

fn extract_feature_points<Keypoint>(
    img: &core::Mat,
) -> opencv::Result<(Vec<crate::Keypoint>, core::Mat)> {
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
        .map(|kp| crate::Keypoint {
            x: kp.pt().x,
            y: kp.pt().y,
            angle: kp.angle(),
        })
        .collect();

    Ok((keypoints, descriptors))
}

fn save_json(data: Vec<crate::Keypoint>) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open("output/serialized_data.json")?;
    let keypoint_list = crate::KeypointList { keypoints: data };
    let json_data = serde_json::to_string(&keypoint_list).unwrap();
    writeln!(&file, "{}", json_data).unwrap();
    Ok(())
}

fn main() {
    let img = imgcodecs::imread("assets/sample_0.png", imgcodecs::IMREAD_GRAYSCALE).unwrap();
    let result = extract_feature_points::<core::KeyPoint>(&img).unwrap();
    save_json(result.0).unwrap();
}
