extern crate opencv;
use opencv::core;
use opencv::imgcodecs;

#[path = "utils/extract_feature_points.rs"]
pub mod extract_feature_points;
use extract_feature_points::extract_feature_points;

#[path = "models/structures.rs"]
pub mod structures;
use structures::Keypoint;

#[path = "utils/save_json.rs"]
pub mod save_json;
use save_json::save_json;

fn main() {
    let img = imgcodecs::imread("assets/sample_0.png", imgcodecs::IMREAD_GRAYSCALE).unwrap();
    let result = extract_feature_points::<core::KeyPoint>(&img).unwrap();
    save_json(result.0).unwrap();
}
