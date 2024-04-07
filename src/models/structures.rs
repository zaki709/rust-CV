use serde;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Descriptor {
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KeypointList {
    pub keypoints: Vec<Keypoint>,
}
