use serde_json;
use std::fs::OpenOptions;
use std::io::Write;

#[path = "../models/structures.rs"]
pub mod structures;

pub fn save_json(data: Vec<crate::Keypoint>) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open("output/serialized_data.json")?;
    let keypoint_list = crate::structures::KeypointList { keypoints: data };
    let json_data = serde_json::to_string(&keypoint_list).unwrap();
    writeln!(&file, "{}", json_data).unwrap();
    Ok(())
}
