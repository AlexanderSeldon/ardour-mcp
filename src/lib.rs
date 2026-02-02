use schemars::JsonSchema; // Ensure necessary imports are present
use serde::{Deserialize, Serialize}; // Ensure necessary imports are present
use std::collections::HashMap;

#[derive(Clone, Debug, JsonSchema, Serialize, Deserialize)]
pub struct TrackInfo {
    pub id: i32,
    pub name: String,
    pub track_type: String, // e.g., "audio", "midi", "bus"
}

#[derive(Clone, Debug, JsonSchema, Serialize, Deserialize)]
pub struct PluginParameter {
    pub param_id: i32,
    pub name: String,
    pub value: f32,  // Normalized 0.0-1.0
}

#[derive(Clone, Debug, JsonSchema, Serialize, Deserialize)]
pub struct PluginInfo {
    pub slot: i32,
    pub name: String,
    pub parameters: HashMap<i32, PluginParameter>, // param_id -> parameter
} 