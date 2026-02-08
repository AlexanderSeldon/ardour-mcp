use anyhow::Result;
use rmcp::{
    model::{
    AnnotateAble,
    CallToolResult,
    Content,
    GetPromptRequestParam,
    GetPromptResult,
    Implementation,
    ListPromptsResult,
    ListResourcesResult,
    PaginatedRequestParam,
    ProtocolVersion,
    RawResource,
    ReadResourceRequestParam,
    ReadResourceResult,
    Resource,
    ServerCapabilities,
    ServerInfo,
        ToolsCapability,
        ResourcesCapability,
    },
    Error as McpError, 
    RoleServer,
    ServerHandler,
    ServiceExt,
    service::RequestContext,
    tool,
    transport::stdio,
};

use nannou_osc as osc;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json; 
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::sync::oneshot;

// Add imports for file logging
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
// Import for combining writers
use tracing_subscriber::fmt::writer::MakeWriterExt;

use ardour_mcp::TrackInfo;

const ARDOUR_OSC_TARGET_ADDR: &str = "127.0.0.1:3819";
#[allow(dead_code)]
const MCP_SERVER_OSC_LISTEN_ADDR: &str = "127.0.0.1:9099";

// Helper to get plugin URI from plugin name (common plugins)
fn get_plugin_uri_from_name(plugin_name: &str) -> Option<String> {
    let name_lower = plugin_name.to_lowercase();
    // Common Calf plugin mappings
    if name_lower.contains("compressor") {
        Some("http://calf.sourceforge.net/plugins/Compressor".to_string())
    } else if name_lower.contains("equalizer") || name_lower.contains("eq") {
        Some("http://calf.sourceforge.net/plugins/Equalizer8Band".to_string())
    } else if name_lower.contains("limiter") {
        Some("http://calf.sourceforge.net/plugins/Limiter".to_string())
    } else if name_lower.contains("reverb") {
        Some("http://calf.sourceforge.net/plugins/Reverb".to_string())
    } else if name_lower.contains("deesser") {
        Some("http://calf.sourceforge.net/plugins/Deesser".to_string())
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
enum PlaybackStatus {
    Playing,
    Stopped,
    Unknown,
}

#[derive(Debug, Clone)]
struct ArdourState {
    playback_status: PlaybackStatus,
    strip_list: Vec<TrackInfo>,
    transport_frame: Option<i64>, // Current playhead position in samples
    // Plugin parameter storage: (ssid, slot, param_id) -> value
    plugin_parameters: HashMap<(i32, i32, i32), f32>,
    // Plugin parameter names: (ssid, slot, param_id) -> name
    plugin_parameter_names: HashMap<(i32, i32, i32), String>,
    // Track which strip and plugin are currently selected (for feedback)
    selected_strip: Option<i32>,
    selected_plugin_slot: Option<i32>,
    // Plugin list cache: ssid -> Vec<(slot, name, enabled)>
    // slot is 0-indexed (piid), name is plugin name, enabled is bool
    plugin_lists: HashMap<i32, Vec<(i32, String, bool)>>,
    // Parameter mapping: (ssid, slot, param_id) -> (is_input, is_control, visible_index)
    // This helps us map param_id to visible index for /select/plugin/parameter
    // visible_index is the 1-indexed position in the list of INPUT parameters only
    // We cache this to avoid querying descriptor repeatedly
    parameter_mapping: HashMap<(i32, i32, i32), (bool, bool, Option<i32>)>,
}

impl ArdourState {
    fn new() -> Self {
        Self {
            playback_status: PlaybackStatus::Unknown,
            strip_list: Vec::new(),
            transport_frame: None,
            plugin_parameters: HashMap::new(),
            plugin_parameter_names: HashMap::new(),
            selected_strip: None,
            selected_plugin_slot: None,
            plugin_lists: HashMap::new(),
            parameter_mapping: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct ArdourService {
    osc_sender: Arc<Mutex<osc::Sender<osc::Connected>>>,
    ardour_state: Arc<Mutex<ArdourState>>,
    pending_requests: Arc<Mutex<PendingRequests>>,
}

#[derive(Debug, Default)]
struct PendingRequests {
    // (ssid, slot, param_id) -> oneshot to deliver value from /strip/plugin/parameter/value
    param_values: HashMap<(i32, i32, i32), oneshot::Sender<f32>>,
    // (ssid, slot) -> oneshot to deliver batch parameter values from /strip/plugin/parameter/values
    batch_param_values: HashMap<(i32, i32), oneshot::Sender<Vec<f32>>>,
    // (ssid, slot) -> oneshot to deliver completion from /strip/plugin/descriptor_end
    descriptor_done: HashMap<(i32, i32), oneshot::Sender<()>>,
    // MixPilot reply channels for native OSC endpoints
    sample_rate: Option<oneshot::Sender<i32>>,
    loop_range: Option<oneshot::Sender<Option<(i32, i32, f32, f32)>>>,
    auto_value: Option<oneshot::Sender<f32>>,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackMuteArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired mute state (true for mute, false for unmute).")]
    mute_state: bool,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTransportSpeedArgs {
    #[schemars(description = "The desired transport speed. Valid range: -8.0 to 8.0.")]
    speed: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct LocateToolArgs {
    #[schemars(description = "The position in samples to locate to.")]
    spos: i64, 
    #[schemars(description = "Whether to start playing after locating. 0 for stop, 1 for play.")]
    roll: i32, 
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackSoloArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired solo state. 0 for solo off, 1 for solo on.")]
    solo_st: i32, 
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackRecEnableArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired record enable state. 0 for off, 1 for on.")]
    rec_st: i32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackGainAbsArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired absolute gain. Valid range: 0.0 to 2.0.")]
    gain_abs: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackGainDBArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired gain in dB. Valid range: -400.0 to 6.0.")]
    gain_db: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackTrimAbsArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired absolute trim. Valid range: 0.1 to 10.0.")]
    trim_abs: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetTrackTrimDBArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The desired trim in dB. Valid range: -20.0 to 20.0.")]
    trim_db: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct AccessActionArgs {
    #[schemars(description = "The name of the Ardour menu action to execute (e.g., 'Editor/zoom-to-session').")]
    action_name: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SelectStripArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus to select.")]
    rid: i32,
    #[schemars(description = "The desired select state (true to select). Currently, only true (1) is effective for selection.")]
    select_state: bool,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripPluginActiveArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The 1-indexed slot of the plugin on the strip.")]
    plugin_slot: i32, 
    #[schemars(description = "The desired activation state (true for active, false for inactive).")]
    active_state: bool,
}

#[derive(Deserialize, JsonSchema, Debug)]
struct InsertStripPluginArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The LV2 URI of the plugin to insert (e.g., 'http://calf.sourceforge.net/plugins/Compressor').")]
    plugin_uri: String,
    #[schemars(description = "Optional slot position (1-indexed). If 0 or omitted, plugin is appended at the end.")]
    slot: Option<i32>,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripPluginParameterArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The 1-indexed slot of the plugin on the strip.")]
    plugin_slot: i32,
    #[schemars(description = "The 1-indexed ID of the parameter within the plugin.")]
    param_id: i32,
    #[schemars(description = "The desired parameter value, normalized (0.0 to 1.0).")]
    value: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SelectPluginArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The 1-indexed slot of the plugin on the strip.")]
    plugin_slot: i32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct GetPluginParametersArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The 1-indexed slot of the plugin on the strip.")]
    plugin_slot: i32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct ListStripPluginsArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct GetMeterReadingsArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus to read meters from.")]
    rid: i32,
}

#[derive(Deserialize, JsonSchema, Debug)]
struct RequestStripListArgs {
    // No arguments needed - just requests the list
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripPanStereoWidthArgs {
    #[schemars(description = "The Router ID (rid) of the stereo track/bus.")]
    rid: i32,
    #[schemars(description = "The desired stereo width. Valid range: 0.0 to 1.0. Default is 1.0 (full width).")]
    width: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetSelectedStripPanStereoWidthArgs {
    #[schemars(description = "The desired stereo width for the currently selected strip. Valid range: 0.0 to 1.0. Default is 1.0 (full width).")]
    width: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripSendGainDbArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The send id (1-based, as used by Ardour OSC /strip/send/*).")]
    send_id: i32,
    #[schemars(description = "Send gain in dB.")]
    gain_db: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripSendFaderArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The send id (1-based, as used by Ardour OSC /strip/send/*).")]
    send_id: i32,
    #[schemars(description = "Send fader value (float, typically 0.0-1.0).")]
    fader: f32,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripSendEnableArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "The send id (1-based, as used by Ardour OSC /strip/send/*).")]
    send_id: i32,
    #[schemars(description = "Enable state (true=enable, false=disable).")]
    enable_state: bool,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[allow(dead_code)]
struct SetStripPolarityArgs {
    #[schemars(description = "The Router ID (rid) of the track/bus.")]
    rid: i32,
    #[schemars(description = "Invert polarity on all channels (true=invert, false=normal).")]
    invert_state: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct GetSessionSampleRateArgs {}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct GetLoopRangeArgs {}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct WritePluginAutomationArgs {
    #[schemars(description = "Route ID (numeric)")]
    route_id: i32,
    #[schemars(description = "Plugin slot (0-indexed)")]
    plugin_slot: i32,
    #[schemars(description = "Parameter control index (0-indexed)")]
    param_id: i32,
    #[schemars(description = "Time position in samples")]
    time_samples: i64,
    #[schemars(description = "Parameter value (0.0-1.0 normalized)")]
    value: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ReadAutomationValueArgs {
    #[schemars(description = "Route ID (numeric)")]
    route_id: i32,
    #[schemars(description = "Control type: 'plugin', 'gain', 'pan_azimuth', or 'pan_width'")]
    control_type: String,
    #[schemars(description = "Plugin slot (0-indexed, required for 'plugin' control_type)")]
    plugin_slot: Option<i32>,
    #[schemars(description = "Parameter control index (0-indexed, required for 'plugin' control_type)")]
    param_id: Option<i32>,
    #[schemars(description = "Time position in samples")]
    time_samples: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct WriteTrackAutomationArgs {
    #[schemars(description = "Route ID (numeric)")]
    route_id: i32,
    #[schemars(description = "Control type: 'gain', 'pan_azimuth', or 'pan_width'")]
    control_type: String,
    #[schemars(description = "Time position in samples")]
    time_samples: i64,
    #[schemars(description = "Parameter value (0.0-1.0 normalized)")]
    value: f32,
    #[schemars(description = "If true, read current value and use it instead of provided value")]
    use_current_as_start: Option<bool>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct SetRegionParameterArgs {
    #[schemars(description = "Route ID (numeric) or name. If name, will be looked up.")]
    route: String,
    #[schemars(description = "Plugin name (for parameter setting)")]
    plugin_name: String,
    #[schemars(description = "Plugin URI (LV2 URI, e.g., 'http://calf.sourceforge.net/plugins/Compressor'). Required for adding plugin.")]
    plugin_uri: Option<String>,
    #[schemars(description = "Parameter name")]
    param_name: String,
    #[schemars(description = "Parameter value (0.0-1.0 normalized)")]
    value: f32,
    #[schemars(description = "Start time in samples")]
    start_samples: i64,
    #[schemars(description = "End time in samples")]
    end_samples: i64,
}

#[tool(tool_box)] 
impl ArdourService {
    pub fn new() -> Result<Self> { 
        tracing::info!("Attempting to create OSC sender for Ardour at {}", ARDOUR_OSC_TARGET_ADDR);
        let sender = osc::sender()
            .map_err(|e| anyhow::anyhow!("Failed to create OSC sender builder: {}", e))?
            .connect(ARDOUR_OSC_TARGET_ADDR)
            .map_err(|e| anyhow::anyhow!("Failed to prepare OSC sender for {}: {}", ARDOUR_OSC_TARGET_ADDR, e))?;
        tracing::info!("OSC sender created and connected to Ardour at {}", ARDOUR_OSC_TARGET_ADDR);
        Ok(Self {
            osc_sender: Arc::new(Mutex::new(sender)),
            ardour_state: Arc::new(Mutex::new(ArdourState::new())),
            pending_requests: Arc::new(Mutex::new(PendingRequests::default())),
        })
    }

    async fn send_osc_setup_to_ardour(&self) -> Result<()> {
        tracing::info!("Sending /set_surface to Ardour to enable OSC feedback.");

        // Extract port from MCP_SERVER_OSC_LISTEN_ADDR ("127.0.0.1:9099")
        let parts: Vec<&str> = MCP_SERVER_OSC_LISTEN_ADDR.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!("Invalid MCP_SERVER_OSC_LISTEN_ADDR format: {}", MCP_SERVER_OSC_LISTEN_ADDR));
        }
        let feedback_port_num: i32 = parts[1].parse()
            .map_err(|e| anyhow::anyhow!("Failed to parse port from MCP_SERVER_OSC_LISTEN_ADDR: {}", e))?;

        tracing::info!("Targeting feedback port: {}", feedback_port_num);

        // Construct the OSC message for /set_surface with all parameters
        // /set_surface i:0 i:159 i:1 i:0 i:0 i:0 i:<feedback_port_num> i:0 i:0
        // According to Ardour manual: /set_surface bank_size strip_types feedback fadermode ...
        // Feedback bits: [0]=buttons, [1]=variable_values (REQUIRED for plugin params!), [2]=ssid_in_path,
        // [3]=heartbeat, [4]=master_section, [5]=bar_and_beat, [6]=smpte, [7]=meter_float, [8]=meter_led,
        // [9]=signal_present, [10]=hp_samples, [11]=hp_min_sec, [12]=hp_gui, [13]=select_fb, [14]=use_osc10,
        // [15]=trigger_status, [16]=scene_status
        // IMPORTANT: We send a very comprehensive feedback value that includes ALL common feedback types
        // This ensures we don't override GUI settings that might have more feedback enabled.
        // The value 126975+ is typical for users with "Strip Controls" enabled in GUI.
        // We ensure bit 1 (variable_values=2) is always set for plugin parameter feedback.
        let feedback_flags = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536; // All feedback types including variable_values (2)
        let osc_args = vec![
            osc::Type::Int(0),    // surface_id (bank_size = 0 for no banking / infinite)
            osc::Type::Int(2047),  // strip_types: Include ALL common strip types (bits 0-10: Audio Tracks, MIDI Tracks, Audio Busses, MIDI Busses, VCAs, Master, Monitor, Foldback, Selected, Hidden, Groups)
            osc::Type::Int(feedback_flags), // feedback (MUST include bit 1=2 for variable_values/plugin params!)
            osc::Type::Int(0),    // fadermode (0 = dB)
            osc::Type::Int(0),    // send_page_size
            osc::Type::Int(0),    // plugin_page_size
            osc::Type::Int(feedback_port_num), // feedback_port
            osc::Type::Int(0),    // linkset (n_strips in old comment)
            osc::Type::Int(0)     // linkid (n_sends_per_strip in old comment)
        ];

        match self.send_osc_message("/set_surface", Some(osc_args)).await {
            Ok(_) => {
                tracing::info!("/set_surface command sent successfully to Ardour.");
                Ok(())
            }
            Err(e) => {
                let err_msg = format!("Failed to send /set_surface OSC message to Ardour: {}", e);
                tracing::error!("{}", err_msg);
                Err(anyhow::anyhow!(err_msg))
            }
        }
    }

    async fn send_osc_message(&self, address: &str, args: Option<Vec<osc::Type>>) -> Result<()> {
        let osc_sender_clone = Arc::clone(&self.osc_sender);
        let owned_address = address.to_string();
        tokio::task::spawn_blocking(move || {
            let sender_guard = osc_sender_clone.blocking_lock();
            let msg_args = args.unwrap_or_default();
            let msg = osc::Message { addr: owned_address, args: msg_args };
            sender_guard.send(msg).map_err(|e| {
                let err_msg = format!("{}", e);
                // Check if this is a connection refused error
                if err_msg.contains("Connection refused") || err_msg.contains("os error 61") {
                    anyhow::anyhow!("OSC connection to Ardour lost: {}. Ardour may have crashed or OSC server stopped.", err_msg)
                } else {
                    anyhow::anyhow!("Failed to send OSC message: {}", e)
                }
            })
        }).await??; 
        Ok(())
    }

    async fn request_plugin_descriptor(&self, ssid: i32, piid: i32) -> Result<()> {
        let mut osc_args = Vec::new();
        osc_args.push(osc::Type::Int(ssid));
        osc_args.push(osc::Type::Int(piid));
        self.send_osc_message("/strip/plugin/descriptor", Some(osc_args)).await
    }

    async fn request_plugin_parameter_value(&self, ssid: i32, piid: i32, param_index: i32) -> Result<()> {
        let mut osc_args = Vec::new();
        osc_args.push(osc::Type::Int(ssid));
        osc_args.push(osc::Type::Int(piid));
        osc_args.push(osc::Type::Int(param_index));
        self.send_osc_message("/strip/plugin/parameter/get", Some(osc_args)).await
    }

    async fn await_descriptor_done(&self, ssid: i32, slot0: i32, timeout_ms: u64) -> Result<()> {
        let (tx, rx) = oneshot::channel::<()>();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.descriptor_done.insert((ssid, slot0), tx);
        }
        tokio::time::timeout(Duration::from_millis(timeout_ms), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timed out waiting for /strip/plugin/descriptor_end for strip {} slot {}", ssid, slot0))?
            .map_err(|_| anyhow::anyhow!("Descriptor wait cancelled for strip {} slot {}", ssid, slot0))?;
        Ok(())
    }

    async fn await_param_value(&self, ssid: i32, slot0: i32, param_index: i32, timeout_ms: u64) -> Result<f32> {
        let (tx, rx) = oneshot::channel::<f32>();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.param_values.insert((ssid, slot0, param_index), tx);
        }
        let v = tokio::time::timeout(Duration::from_millis(timeout_ms), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timed out waiting for /strip/plugin/parameter/value strip {} slot {} param {}", ssid, slot0, param_index))?
            .map_err(|_| anyhow::anyhow!("Param wait cancelled strip {} slot {} param {}", ssid, slot0, param_index))?;
        Ok(v)
    }

    async fn request_plugin_parameters_batch(&self, ssid: i32, piid: i32) -> Result<()> {
        let mut osc_args = Vec::new();
        osc_args.push(osc::Type::Int(ssid));
        osc_args.push(osc::Type::Int(piid));
        self.send_osc_message("/strip/plugin/parameter/get_all", Some(osc_args)).await
    }

    async fn await_batch_param_values(&self, ssid: i32, slot0: i32, timeout_ms: u64) -> Result<Vec<f32>> {
        let (tx, rx) = oneshot::channel::<Vec<f32>>();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.batch_param_values.insert((ssid, slot0), tx);
        }
        let values = tokio::time::timeout(Duration::from_millis(timeout_ms), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timed out waiting for /strip/plugin/parameter/values strip {} slot {}", ssid, slot0))?
            .map_err(|_| anyhow::anyhow!("Batch param wait cancelled strip {} slot {}", ssid, slot0))?;
        Ok(values)
    }

    // MixPilot await helpers for native OSC reply endpoints
    async fn await_sample_rate(&self, timeout_ms: u64) -> Result<i32> {
        let (tx, rx) = oneshot::channel::<i32>();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.sample_rate = Some(tx);
        }
        let v = tokio::time::timeout(Duration::from_millis(timeout_ms), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timed out waiting for /mixpilot/reply/sample_rate"))?
            .map_err(|_| anyhow::anyhow!("Sample rate request cancelled"))?;
        Ok(v)
    }

    async fn await_loop_range(&self, timeout_ms: u64) -> Result<Option<(i32, i32, f32, f32)>> {
        let (tx, rx) = oneshot::channel::<Option<(i32, i32, f32, f32)>>();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.loop_range = Some(tx);
        }
        let v = tokio::time::timeout(Duration::from_millis(timeout_ms), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timed out waiting for /mixpilot/reply/loop_range"))?
            .map_err(|_| anyhow::anyhow!("Loop range request cancelled"))?;
        Ok(v)
    }

    async fn await_auto_value(&self, timeout_ms: u64) -> Result<f32> {
        let (tx, rx) = oneshot::channel::<f32>();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.auto_value = Some(tx);
        }
        let v = tokio::time::timeout(Duration::from_millis(timeout_ms), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Timed out waiting for /mixpilot/reply/auto_value"))?
            .map_err(|_| anyhow::anyhow!("Auto value request cancelled"))?;
        Ok(v)
    }

    #[tool(name = "transport_play", description = "Starts Ardour playback.")]
    async fn transport_play_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing transport_play_tool");
        match self.send_osc_message("/transport_play", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Playback started")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "transport_stop", description = "Stops Ardour playback.")]
    async fn transport_stop_tool(&self) -> Result<CallToolResult, McpError> { 
        tracing::info!("Executing transport_stop_tool");
        match self.send_osc_message("/transport_stop", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Playback stopped")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "get_transport_state", description = "Gets the current transport state (Playing, Stopped, or Unknown). Requests refresh from Ardour if state is Unknown.")]
    async fn get_transport_state_tool(&self) -> Result<CallToolResult, McpError> {
        // #region agent log
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"TRANSPORT_ENTRY","location":"main.rs:433","message":"get_transport_state_tool: ENTRY","data":{{}},"timestamp":{}}}"#, timestamp);
        }
        // #endregion
        tracing::info!("Executing get_transport_state_tool");
        
        // Check current state
        let state = self.ardour_state.lock().await;
        let current_status = state.playback_status.clone();
        let frame = state.transport_frame.map(|f| f.to_string()).unwrap_or_else(|| "Unknown".to_string());
        drop(state); // Release lock before async operation
        
        // If state is Unknown, try to request refresh from Ardour
        // Ardour doesn't have a direct query command, but we can try sending a harmless command
        // that might trigger feedback, or just return Unknown if connection is lost
        let status_str = match current_status {
            PlaybackStatus::Playing => "Playing".to_string(),
            PlaybackStatus::Stopped => "Stopped".to_string(),
            PlaybackStatus::Unknown => {
                // Try to request transport frame update which might trigger state feedback
                // This is a read-only operation that shouldn't cause issues
                if let Err(e) = self.send_osc_message("/transport_frame", None).await {
                    tracing::warn!("Could not request transport state refresh from Ardour: {}. State remains Unknown.", e);
                    // If connection is refused, this indicates Ardour is not responding
                    let err_str = format!("{}", e);
                    if err_str.contains("Connection refused") || err_str.contains("os error 61") {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "OSC connection to Ardour lost. Ardour may have crashed or OSC server stopped."
                        )]));
                    }
                    "Unknown".to_string()
                } else {
                    // Give Ardour a moment to send feedback
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    // Re-check state after potential feedback
                    let state_after = self.ardour_state.lock().await;
                    match state_after.playback_status {
                        PlaybackStatus::Playing => "Playing".to_string(),
                        PlaybackStatus::Stopped => "Stopped".to_string(),
                        PlaybackStatus::Unknown => "Unknown".to_string(),
                    }
                }
            }
        };
        
        let result_json = json!({
            "status": status_str,
            "transport_frame": frame
        });
        
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result_json)
                .unwrap_or_else(|_| "Failed to serialize transport state".to_string())
        )]))
    }

    #[tool(name = "goto_start", description = "Moves the playhead to the session start.")]
    async fn goto_start_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing goto_start_tool");
        match self.send_osc_message("/goto_start", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Playhead moved to start")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "goto_end", description = "Moves the playhead to the session end.")]
    async fn goto_end_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing goto_end_tool");
        match self.send_osc_message("/goto_end", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Playhead moved to end")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "loop_toggle", description = "Toggles loop playback mode.")]
    async fn loop_toggle_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing loop_toggle_tool");
        match self.send_osc_message("/loop_toggle", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Loop mode toggled")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "undo", description = "Undoes the last action.")]
    async fn undo_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing undo_tool");
        match self.send_osc_message("/undo", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Undo action performed")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "redo", description = "Redoes the last undone action.")]
    async fn redo_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing redo_tool");
        match self.send_osc_message("/redo", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Redo action performed")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "toggle_punch_in", description = "Toggles the Punch In state.")]
    async fn toggle_punch_in_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing toggle_punch_in_tool");
        match self.send_osc_message("/toggle_punch_in", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Punch In toggled")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "toggle_punch_out", description = "Toggles the Punch Out state.")]
    async fn toggle_punch_out_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing toggle_punch_out_tool");
        match self.send_osc_message("/toggle_punch_out", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Punch Out toggled")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "rec_enable_toggle", description = "Toggles the master record enable or selected track record enable.")]
    async fn rec_enable_toggle_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing rec_enable_toggle_tool");
        match self.send_osc_message("/rec_enable_toggle", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Record Enable toggled")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "toggle_all_rec_enables", description = "Toggles the record enable state for ALL tracks.")]
    async fn toggle_all_rec_enables_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing toggle_all_rec_enables_tool");
        match self.send_osc_message("/toggle_all_rec_enables", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("All Record Enables toggled")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "ffwd", description = "Fast forwards the transport.")]
    async fn ffwd_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing ffwd_tool");
        match self.send_osc_message("/ffwd", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Fast Forward activated")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "rewind", description = "Rewinds the transport.")]
    async fn rewind_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing rewind_tool");
        match self.send_osc_message("/rewind", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Rewind activated")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "add_marker", description = "Adds a location marker at the current playhead position.")]
    async fn add_marker_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing add_marker_tool");
        match self.send_osc_message("/add_marker", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Marker added")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "next_marker", description = "Moves the playhead to the next location marker.")]
    async fn next_marker_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing next_marker_tool");
        match self.send_osc_message("/next_marker", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Moved to next marker")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "prev_marker", description = "Moves the playhead to the previous location marker.")]
    async fn prev_marker_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing prev_marker_tool");
        match self.send_osc_message("/prev_marker", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Moved to previous marker")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "save_state", description = "Saves the current session state.")]
    async fn save_state_tool(&self) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing save_state_tool");
        match self.send_osc_message("/save_state", None).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text("Session state saved")])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("OSC send error: {}",e))])),
        }
    }

    #[tool(name = "set_track_mute", description = "Sets the mute state of a specific track.")]
    async fn set_track_mute_tool(
        &self,
        #[schemars(description = "Arguments for setting track mute state. Requires 'rid' (integer) and 'mute_state' (boolean).")]
        #[tool(aggr)] args: SetTrackMuteArgs 
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_mute_tool with args: {:?}", args);

        let osc_mute_state = if args.mute_state { 1i32 } else { 0i32 };
        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Int(osc_mute_state)];
        
        match self.send_osc_message("/strip/mute", Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Track {} mute state set to {}",
                args.rid, args.mute_state
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for /strip/mute: {}",
                e
            ))])),
        }
    }

    #[tool(name = "set_transport_speed", description = "Sets Ardour's transport speed. Valid range: -8.0 to 8.0.")]
    async fn set_transport_speed_tool(
        &self,
        #[schemars(description = "Argument for setting transport speed. Requires 'speed' (float).")]
        #[tool(aggr)] args: SetTransportSpeedArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_transport_speed_tool with speed: {}",
            args.speed
        );

        if args.speed < -8.0 || args.speed > 8.0 {
            tracing::warn!("Invalid transport speed: {}. Must be between -8.0 and 8.0.", args.speed);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid transport speed: {}. Must be between -8.0 and 8.0.",
                args.speed
            ))]));
        }

        let osc_args = vec![osc::Type::Float(args.speed)];
        match self.send_osc_message("/set_transport_speed", Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Transport speed set to {}",
                args.speed
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for /set_transport_speed: {}",
                e
            ))])),
        }
    }

    #[tool(name = "locate", description = "Locates the playhead to a specific sample position and optionally starts playback.")]
    async fn locate_tool(
        &self,
        #[schemars(description = "Arguments for locating the playhead. Requires 'spos' (integer samples) and 'roll' (integer 0 or 1).")]
        #[tool(aggr)] args: LocateToolArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing locate_tool with spos: {}, roll: {}", args.spos, args.roll);
        let osc_args = vec![osc::Type::Long(args.spos), osc::Type::Int(args.roll)];
        match self.send_osc_message("/locate", Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Located to sample {} with roll state {}",
                args.spos, args.roll
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for /locate: {}",
                e
            ))])),
        }
    }

    #[tool(name = "set_track_solo", description = "Sets the solo state of a specific track.")]
    async fn set_track_solo_tool(
        &self,
        #[schemars(description = "Arguments for setting track solo state. Requires 'rid' (integer) and 'solo_st' (integer: 0 or 1).")]
        #[tool(aggr)] args: SetTrackSoloArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_solo_tool for rid: {}, solo_state: {}", args.rid, args.solo_st);

        if !(args.solo_st == 0 || args.solo_st == 1) {
            tracing::warn!("Invalid solo_st value: {}. Must be 0 or 1.", args.solo_st);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid solo_st value: {}. Must be 0 (off) or 1 (on).",
                args.solo_st
            ))]));
        }

        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Int(args.solo_st)];
        let address = "/strip/solo"; 
        
        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Track {} solo state set to {}",
                args.rid, args.solo_st
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_track_rec_enable", description = "Sets the record enable state of a specific track.")]
    async fn set_track_rec_enable_tool(
        &self,
        #[schemars(description = "Arguments for setting track record enable state. Requires 'rid' (integer) and 'rec_st' (integer: 0 or 1).")]
        #[tool(aggr)] args: SetTrackRecEnableArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_rec_enable_tool for rid: {}, rec_enable_state: {}", args.rid, args.rec_st);

        if !(args.rec_st == 0 || args.rec_st == 1) {
            tracing::warn!("Invalid rec_st value: {}. Must be 0 or 1.", args.rec_st);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid rec_st value: {}. Must be 0 (off) or 1 (on).",
                args.rec_st
            ))]));
        }

        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Int(args.rec_st)];
        let address = "/strip/recenable"; 
        
        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Track {} record enable state set to {}",
                args.rid, args.rec_st
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_track_gain_abs", description = "Sets the absolute gain of a specific track.")]
    async fn set_track_gain_abs_tool(
        &self,
        #[schemars(description = "Arguments for setting track absolute gain. Requires 'rid' (integer) and 'gain_abs' (float: 0.0 to 2.0). Maps to fader 0.0-1.0.")]
        #[tool(aggr)] args: SetTrackGainAbsArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_gain_abs_tool for rid: {}, gain_abs: {}", args.rid, args.gain_abs);

        
        if args.gain_abs < 0.0 || args.gain_abs > 2.0 {
            tracing::warn!("Invalid gain_abs value: {}. Must be between 0.0 and 2.0 (maps to fader 0.0-1.0).", args.gain_abs);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid gain_abs value: {}. Must be between 0.0 and 2.0 (maps to fader 0.0-1.0).",
                args.gain_abs
            ))]));
        }
        let fader_position = (args.gain_abs / 2.0).clamp(0.0, 1.0);

        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Float(fader_position)];
        let address = "/strip/fader"; 
        
        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Track {} fader position set to {} (from gain_abs {})",
                args.rid, fader_position, args.gain_abs
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_track_gain_db", description = "Sets the gain of a specific track in dB.")]
    async fn set_track_gain_db_tool(
        &self,
        #[schemars(description = "Arguments for setting track gain in dB. Requires 'rid' (integer) and 'gain_db' (float: -400.0 to 6.0).")]
        #[tool(aggr)] args: SetTrackGainDBArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_gain_db_tool for rid: {}, gain_db: {}", args.rid, args.gain_db);

        if !(args.gain_db >= -400.0 && args.gain_db <= 6.0) {
            tracing::warn!("Invalid gain_db value: {}. Must be between -400.0 and 6.0.", args.gain_db);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid gain_db value: {}. Must be between -400.0 and 6.0.",
                args.gain_db
            ))]));
        }

        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Float(args.gain_db)];
        let address = "/strip/gain"; 
        
        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Track {} gain (dB) set to {}",
                args.rid, args.gain_db
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_track_trim_abs", description = "Sets the absolute trim of a specific track.")]
    async fn set_track_trim_abs_tool(
        &self,
        #[schemars(description = "Arguments for setting track absolute trim. Requires 'rid' (integer) and 'trim_abs' (float: 0.1 to 10.0). Maps to trim_fader 0.0-1.0.")]
        #[tool(aggr)] args: SetTrackTrimAbsArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_trim_abs_tool for rid: {}, trim_abs: {}", args.rid, args.trim_abs);

        
        if args.trim_abs < 0.1 || args.trim_abs > 10.0 {
            tracing::warn!("Invalid trim_abs value: {}. Must be between 0.1 and 10.0.", args.trim_abs);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid trim_abs value: {}. Must be between 0.1 and 10.0.",
                args.trim_abs
            ))]));
        }
        let fader_position = ((args.trim_abs - 0.1) / 9.9).clamp(0.0, 1.0);

        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Float(fader_position)];
        let address = "/strip/trim_fader"; 
        
        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Track {} trim fader position set to {} (from trim_abs {})",
                args.rid, fader_position, args.trim_abs
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_track_trim_db", description = "Sets the trim of a specific track in dB.")]
    async fn set_track_trim_db_tool(
        &self,
        #[schemars(description = "Arguments for setting track trim in dB. Requires 'rid' (integer) and 'trim_db' (float: -20.0 to 20.0).")]
        #[tool(aggr)] args: SetTrackTrimDBArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing set_track_trim_db_tool for rid: {}, trim_db: {}", args.rid, args.trim_db);

        if args.trim_db < -20.0 || args.trim_db > 20.0 {
            let error_msg = format!(
                "Invalid trim_db value: {}. Must be between -20.0 and 20.0.",
                args.trim_db
            );
            tracing::warn!("{}", error_msg);
            return Ok(CallToolResult::error(vec![Content::text(error_msg)]));
        }

        let osc_addr = "/strip/trimdB"; 
        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Float(args.trim_db)];

        match self.send_osc_message(osc_addr, Some(osc_args)).await {
            Ok(_) => {
                let success_msg = format!(
                    "Successfully sent OSC message {} with rid {} and trim_db {}",
                    osc_addr, args.rid, args.trim_db
                );
                tracing::info!("{}", success_msg);
                Ok(CallToolResult::success(vec![Content::text(success_msg)]))
            }
            Err(e) => {
                let error_msg = format!("Failed to send OSC message {} for rid {}: {:?}", osc_addr, args.rid, e);
                tracing::error!("{}", error_msg);
                Err(McpError::internal_error(error_msg, None))
            }
        }
    }

    #[tool(name = "access_action", description = "Executes a specified Ardour menu action by its name.")]
    async fn access_action_tool(
        &self,
        #[schemars(description = "Argument for accessing menu action. Requires 'action_name' (string).")]
        #[tool(aggr)] args: AccessActionArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing access_action_tool for action_name: {}", args.action_name);

        if args.action_name.is_empty() {
            let error_msg = "action_name cannot be empty.";
            tracing::warn!("{}", error_msg);
            return Ok(CallToolResult::error(vec![Content::text(error_msg.to_string())]));
        }

        let osc_addr = "/access_action"; 
        
        let osc_args = vec![osc::Type::String(args.action_name.clone())];

        match self.send_osc_message(osc_addr, Some(osc_args)).await {
            Ok(_) => {
                let success_msg = format!(
                    "Successfully sent OSC message {} with action_name '{}'",
                    osc_addr, args.action_name
                );
                tracing::info!("{}", success_msg);
                Ok(CallToolResult::success(vec![Content::text(success_msg)]))
            }
            Err(e) => {
                let error_msg = format!(
                    "Failed to send OSC message {} for action_name '{}': {:?}",
                    osc_addr, args.action_name, e
                );
                tracing::error!("{}", error_msg);
                Err(McpError::internal_error(error_msg, None))
            }
        }
    }

    #[tool(name = "select_strip", description = "Selects a specific strip (track/bus) in Ardour.")]
    async fn select_strip_tool(
        &self,
        #[schemars(description = "Arguments for selecting a strip. Requires 'rid' (integer) and 'select_state' (boolean, true to select).")]
        #[tool(aggr)] args: SelectStripArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing select_strip_tool for rid: {}, select_state: {}", args.rid, args.select_state);

        // According to Ardour OSC docs for /strip/select, the second arg (y/n) is 1 for select, and 0 is ignored.
        // So, we only send if select_state is true.
        if !args.select_state {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} not selected as select_state was false.",
                args.rid
            ))]));
        }

        if args.rid <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                format!("Invalid rid: {}. Must be a positive integer.", args.rid)
            )]));
        }

        let osc_args = vec![osc::Type::Int(args.rid), osc::Type::Int(1)]; // Always send 1 to select
        let address = "/strip/select"; 
        
        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => {
                // Update state immediately (don't wait for feedback)
                // IMPORTANT: Preserve plugin list cache when selecting strip - don't clear it
                // The cache might be needed immediately after for select_plugin
                let mut state = self.ardour_state.lock().await;
                state.selected_strip = Some(args.rid);
                state.selected_plugin_slot = None; // Clear plugin selection when strip changes
                // DO NOT clear plugin_lists - preserve the cache for this strip
                tracing::info!("Strip {} selected and state updated (preserved plugin list cache)", args.rid);
                
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Strip {} selected successfully.",
                    args.rid
                ))]))
            }
            Err(e) => {
                let error_msg = format!("OSC send error for {}: {}", address, e);
                tracing::error!("{}", error_msg);
                Ok(CallToolResult::error(vec![Content::text(error_msg)]))
            }
        }
    }

    #[tool(name = "set_strip_plugin_active", description = "Activates or deactivates a plugin on a specific strip slot.")]
    async fn set_strip_plugin_active_tool(
        &self,
        #[schemars(description = "Arguments for setting plugin activation state. Requires 'rid' (strip ID), 'plugin_slot' (1-indexed), and 'active_state' (boolean).")]
        #[tool(aggr)] args: SetStripPluginActiveArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_plugin_active_tool for rid: {}, plugin_slot: {}, active_state: {}",
            args.rid, args.plugin_slot, args.active_state
        );

        if args.plugin_slot <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid plugin_slot: must be a positive integer.".to_string()
            )]));
        }

        let address = if args.active_state {
            "/strip/plugin/activate"
        } else {
            "/strip/plugin/deactivate"
        };

        let osc_args = vec![
            osc::Type::Int(args.rid),
            osc::Type::Int(args.plugin_slot),
        ];

        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} plugin in slot {} command ({}) sent.",
                args.rid, args.plugin_slot, if args.active_state { "activate" } else { "deactivate" }
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "insert_strip_plugin", description = "Inserts a plugin into a specific strip at the specified slot position.")]
    async fn insert_strip_plugin_tool(
        &self,
        #[schemars(description = "Arguments for inserting a plugin. Requires 'rid' (strip ID), 'plugin_uri' (LV2 URI), and optional 'slot' (1-indexed, 0 for auto-append).")]
        #[tool(aggr)] args: InsertStripPluginArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing insert_strip_plugin_tool for rid: {}, plugin_uri: {}, slot: {:?}",
            args.rid, args.plugin_uri, args.slot
        );

        if args.rid <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid rid: must be a positive integer.".to_string()
            )]));
        }

        let slot_value = args.slot.unwrap_or(0); // 0 = auto-append

        let osc_args = vec![
            osc::Type::Int(args.rid),
            osc::Type::String(args.plugin_uri.clone()),
            osc::Type::Int(slot_value),
        ];

        match self.send_osc_message("/strip/plugin/insert", Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Plugin insertion command sent: {} on strip {} at slot {}",
                args.plugin_uri, args.rid, if slot_value == 0 { "auto-append".to_string() } else { slot_value.to_string() }
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for /strip/plugin/insert: {}",
                e
            ))])),
        }
    }

    #[tool(name = "set_strip_plugin_parameter", description = "Sets a specific parameter of a plugin on a strip. Tries /strip/plugin/parameter first, falls back to /select/plugin/parameter if needed.")]
    async fn set_strip_plugin_parameter_tool(
        &self,
        #[schemars(description = "Arguments for setting a plugin parameter. Requires 'rid', 'plugin_slot', 'param_id', and 'value' (0.0-1.0).")]
        #[tool(aggr)] args: SetStripPluginParameterArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_plugin_parameter_tool for rid: {}, plugin_slot: {}, param_id: {}, value: {}",
            args.rid, args.plugin_slot, args.param_id, args.value
        );

        if args.plugin_slot <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid plugin_slot: must be a positive integer.".to_string()
            )]));
        }
        if args.param_id <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid param_id: must be a positive integer.".to_string()
            )]));
        }
        if !(0.0..=1.0).contains(&args.value) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid value: must be between 0.0 and 1.0 (inclusive).".to_string()
            )]));
        }

        // IMPORTANT:
        // MixPilot inventory (and Ardour's /select/plugin/parameter feedback) uses the *visible parameter index*
        // for /select/plugin/parameter (1..N). This is NOT the same numbering as /strip/plugin/parameter.
        //
        // Ardour also may accept /strip/plugin/parameter OSC packets but reject them internally with:
        // "is not a control input", and there is no OSC error/ack to detect that case reliably.
        //
        // Therefore we always use the selection-based path and treat args.param_id as the visible parameter index.
        tracing::info!("Setting parameter via selection path: /strip/select + /select/plugin + /select/plugin/parameter (visible param index)");
        
        // Step 1: Select the strip
        let select_strip_args = vec![osc::Type::Int(args.rid), osc::Type::Int(1)];
        if let Err(e) = self.send_osc_message("/strip/select", Some(select_strip_args)).await {
            tracing::warn!("Failed to select strip {} for selection path: {}", args.rid, e);
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to set parameter: strip selection failed: {}",
                e
            ))]));
        }

        // Step 2: Calculate visible plugin index (same logic as select_plugin_tool)
        let target_slot_0_indexed = args.plugin_slot - 1;
        let visible_plugin_index = {
            let state = self.ardour_state.lock().await;
            if let Some(plugin_list) = state.plugin_lists.get(&args.rid) {
                let enabled_plugins: Vec<_> = plugin_list.iter()
                    .filter(|(_, _, enabled)| *enabled)
                    .map(|(slot, name, _)| (*slot, name.clone()))
                    .collect();
                
                let pos = enabled_plugins.iter()
                    .position(|(slot, _)| *slot == target_slot_0_indexed);
                
                if let Some(index) = pos {
                    Some((index + 1) as i32)
                } else {
                    // Fallback: use slot number directly
                    Some(args.plugin_slot)
                }
            } else {
                Some(args.plugin_slot)
            }
        };

        // Step 3: Select the plugin
        if let Some(visible_idx) = visible_plugin_index {
            let plugin_select_args = vec![osc::Type::Float(visible_idx as f32)];
            if let Err(e) = self.send_osc_message("/select/plugin", Some(plugin_select_args)).await {
                tracing::warn!("Failed to select plugin for fallback: {}", e);
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to set parameter: /strip/plugin/parameter failed and fallback plugin selection failed: {}",
                    e
                ))]));
            }
            
            // Wait briefly for plugin selection to take effect
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Step 4: Send /select/plugin/parameter with visible parameter index
        let visible_param_index = args.param_id;
        let select_param_args = vec![
            osc::Type::Int(visible_param_index),
            osc::Type::Float(args.value),
        ];
        
        // Note: We can't verify if Ardour accepted the parameter via OSC response
        // The user will see in Ardour logs if it failed with "not a control input"
        match self.send_osc_message("/select/plugin/parameter", Some(select_param_args)).await {
            Ok(_) => {
                tracing::info!("Sent /select/plugin/parameter with visible index {} (param_id {})", visible_param_index, args.param_id);
                // We assume it worked - user will see in logs if it didn't
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Strip {} plugin slot {} parameter {} set to {} (via /select/plugin/parameter, visible index {}).",
                    args.rid, args.plugin_slot, args.param_id, args.value, visible_param_index
                ))]))
            }
            Err(e) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to set parameter via /select/plugin/parameter: {}. Parameter {} may not be controllable via OSC.",
                    e, args.param_id
                ))]))
            }
        }
    }

    #[tool(name = "select_plugin", description = "Selects a plugin on a strip, which triggers Ardour to send parameter feedback.")]
    async fn select_plugin_tool(
        &self,
        #[schemars(description = "Arguments for selecting a plugin. Requires 'rid' and 'plugin_slot'.")]
        #[tool(aggr)] args: SelectPluginArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Executing select_plugin_tool for rid: {}, plugin_slot: {}", args.rid, args.plugin_slot);

        if args.plugin_slot <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid plugin_slot: must be a positive integer.".to_string()
            )]));
        }

        // Refresh plugin list for this strip so the visible index calculation is up-to-date
        if let Err(e) = self.send_osc_message("/strip/plugin/list", Some(vec![osc::Type::Int(args.rid)])).await {
            tracing::warn!("Failed to refresh plugin list for strip {}: {}", args.rid, e);
        } else {
            // Wait briefly for Ardour to respond and populate the cache
            for attempt in 0..4 {
                tokio::time::sleep(Duration::from_millis(150)).await;
                let state = self.ardour_state.lock().await;
                if let Some(plugin_list) = state.plugin_lists.get(&args.rid) {
                    if !plugin_list.is_empty() {
                        tracing::debug!(
                            "Plugin list refreshed for strip {} on attempt {} ({} plugins)",
                            args.rid,
                            attempt + 1,
                            plugin_list.len()
                        );
                        break;
                    }
                }
            }
        }

        // First, select the strip
        let select_strip_args = vec![osc::Type::Int(args.rid), osc::Type::Int(1)];
        match self.send_osc_message("/strip/select", Some(select_strip_args)).await {
            Ok(_) => {
                tracing::debug!("Strip {} selected successfully", args.rid);
            }
            Err(e) => {
                let error_msg = format!("Failed to select strip {}: {}", args.rid, e);
                tracing::error!("{}", error_msg);
                return Ok(CallToolResult::error(vec![Content::text(error_msg)]));
            }
        }

        // Update selected state
        {
            let mut state = self.ardour_state.lock().await;
            state.selected_strip = Some(args.rid);
            state.selected_plugin_slot = Some(args.plugin_slot);
            tracing::info!("Updated selected state: strip={}, plugin_slot={}", args.rid, args.plugin_slot);
        }

        // Select the plugin (this triggers Ardour to send parameter feedback)
        // According to Ardour OSC docs and source code:
        // - /ardour/select/insert takes (RouteID, InsertIX) where InsertIX is actual slot number
        // - /select/plugin takes a Float representing the visible plugin index (1-indexed)
        // Visible index = position in the list of ENABLED plugins only
        // So if slot 1 is disabled and slot 2 is enabled, slot 2 has visible index 1
        // /ardour/select/insert is not supported (Ardour logs show "Unhandled")
        // So we only use /select/plugin, which uses visible plugin index
        
        // Calculate visible index from enabled plugins
        // plugin_slot is 1-indexed (from API), but we store as 0-indexed internally
        let target_slot_0_indexed = args.plugin_slot - 1;
        let visible_index = {
            let state = self.ardour_state.lock().await;
            tracing::warn!("[SELECT_PLUGIN] Looking up plugin list for strip {} (cache has {} strips: {:?})", 
                         args.rid, state.plugin_lists.len(), 
                         state.plugin_lists.keys().collect::<Vec<_>>());
            if let Some(plugin_list) = state.plugin_lists.get(&args.rid) {
                tracing::warn!("[SELECT_PLUGIN] Found plugin list for strip {}: {} plugins total", args.rid, plugin_list.len());
                
                // WORKAROUND: If plugin list is empty but we're trying to select a plugin,
                // use the slot number directly as fallback (cache might have been cleared/overwritten)
                if plugin_list.is_empty() {
                    tracing::warn!("[SELECT_PLUGIN] Plugin list for strip {} is empty (cache may have been cleared). Using slot {} directly as visible index (fallback).", 
                                 args.rid, args.plugin_slot);
                    Some(args.plugin_slot)
                } else {
                    // Log full plugin list for debugging
                    for (slot, name, enabled) in plugin_list {
                        tracing::warn!("[SELECT_PLUGIN]   Plugin list entry: slot {} (0-indexed), name: {}, enabled: {}", slot, name, enabled);
                    }
                    // Filter to only enabled plugins, sorted by slot
                    // Debug: log what we're filtering
                    tracing::warn!("[SELECT_PLUGIN] Filtering {} plugins for enabled ones...", plugin_list.len());
                    for (slot, name, enabled) in plugin_list.iter() {
                        tracing::warn!("[SELECT_PLUGIN]   Plugin: slot {} (0-indexed), name: {}, enabled: {} (bool: {})", 
                                     slot, name, enabled, *enabled);
                    }
                    let mut enabled_plugins: Vec<_> = plugin_list.iter()
                        .filter(|(_, _, enabled)| {
                            let result = *enabled;
                            tracing::warn!("[SELECT_PLUGIN]   Filter check: enabled={}", result);
                            result
                        })
                        .map(|(slot, name, _)| (*slot, name.clone()))
                        .collect();
                    enabled_plugins.sort_by_key(|(slot, _)| *slot);
                    tracing::warn!("[SELECT_PLUGIN] After filtering, found {} enabled plugins", enabled_plugins.len());
                    
                    tracing::warn!("[SELECT_PLUGIN] Strip {} has {} enabled plugins after filtering: {:?}", args.rid, enabled_plugins.len(),
                                  enabled_plugins.iter().map(|(s, n)| format!("slot {}: {}", s + 1, n)).collect::<Vec<_>>());
                    
                    // Find the position of target slot in enabled list (1-indexed)
                    let pos = enabled_plugins.iter()
                        .position(|(slot, _)| *slot == target_slot_0_indexed);
                    
                    if let Some(index) = pos {
                        // Found it - visible index is 1-indexed
                        tracing::info!("Found plugin slot {} (0-indexed: {}) at position {} in enabled list, visible index = {}", 
                                      args.plugin_slot, target_slot_0_indexed, index, index + 1);
                        Some((index + 1) as i32)
                    } else {
                        // Plugin not found in enabled list - might be disabled or missing from incomplete plugin list
                        tracing::warn!("Plugin slot {} (0-indexed: {}) not found in enabled plugins list for strip {}. Available enabled slots: {:?}", 
                                      args.plugin_slot, target_slot_0_indexed, args.rid, 
                                      enabled_plugins.iter().map(|(s, _)| s + 1).collect::<Vec<_>>());
                        // Also log the full plugin list for debugging
                        tracing::warn!("Full plugin list for strip {}: {:?}", args.rid,
                                      plugin_list.iter().map(|(s, n, e)| format!("slot {}: {} (enabled: {})", s + 1, n, e)).collect::<Vec<_>>());
                        
                        // Check if the target slot exists in the full plugin list
                        if let Some((_, name, enabled)) = plugin_list.iter().find(|(slot, _, _)| *slot == target_slot_0_indexed) {
                            if *enabled {
                                // Slot exists and is enabled, but wasn't in filtered enabled_plugins list
                                // This can happen if the plugin list is incomplete or there's a filtering issue
                                // Use slot number directly as fallback - Ardour should handle it
                                tracing::warn!("WORKAROUND: Slot {} ({}) exists and is enabled but not in filtered enabled list. Using slot {} directly as visible index (fallback).", 
                                             target_slot_0_indexed, name, args.plugin_slot);
                                Some(args.plugin_slot)
                            } else {
                                // Slot exists but is disabled - cannot select
                                tracing::warn!("Plugin slot {} ({}) exists but is disabled. Cannot select.", 
                                             args.plugin_slot, name);
                                None
                            }
                        } else {
                            // Slot doesn't exist in plugin list at all - plugin list might be incomplete
                            // This is the key issue: if Ardour's plugin list is incomplete (missing slot 2),
                            // we should still try to select it using the slot number directly
                            // This handles cases where the scan knows about a plugin but Ardour's list doesn't include it
                            tracing::warn!("WORKAROUND: Slot {} not in plugin list (list may be incomplete). Using slot {} directly as visible index (fallback).", 
                                         target_slot_0_indexed, args.plugin_slot);
                            Some(args.plugin_slot)
                        }
                    }
                }
            } else {
                // No plugin list available - fall back to using slot number as visible index
                // (assumes all plugins are enabled and sequential)
                // This is a workaround for when the plugin list hasn't been cached yet or was cleared
                tracing::warn!("No plugin list for strip {} in cache. Available strips in cache: {:?}. Using slot {} directly as visible index (fallback)", 
                             args.rid, state.plugin_lists.keys().collect::<Vec<_>>(), args.plugin_slot);
                Some(args.plugin_slot)
            }
        };
        
        let visible_index = match visible_index {
            Some(idx) => idx,
            None => {
                // Plugin is disabled or not found - return error
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Plugin slot {} on strip {} is disabled or not found. Cannot select disabled plugins.",
                    args.plugin_slot, args.rid
                ))]));
            }
        };
        
        let plugin_select_args_select = vec![osc::Type::Float(visible_index as f32)];
        tracing::info!("Sending /select/plugin with visible_index={} (for actual slot {}, strip {})", 
                      visible_index, args.plugin_slot, args.rid);
        let select_result = self.send_osc_message("/select/plugin", Some(plugin_select_args_select)).await;
        
        match select_result {
               Ok(_) => {
                   tracing::info!("Plugin selection command sent. Waiting for parameter feedback...");
                   // Give Ardour more time to send feedback - parameter messages can take a moment
                   tokio::time::sleep(Duration::from_millis(800)).await;
                   
                   Ok(CallToolResult::success(vec![Content::text(format!(
                       "Plugin on strip {} slot {} selected. Parameter feedback should arrive shortly.",
                       args.rid, args.plugin_slot
                   ))]))
               }
               Err(e) => {
                   let error_msg = format!("Failed to select plugin: {}. Strip {} is selected, but plugin selection failed.", e, args.rid);
                   tracing::warn!("{}", error_msg);
                   // Still return success since strip is selected - plugin selection might work via feedback
                   Ok(CallToolResult::success(vec![Content::text(format!(
                       "Strip {} selected. Plugin selection command failed, but parameters may still arrive via feedback.",
                       args.rid
                   ))]))
               }
           }
    }

    #[tool(name = "get_plugin_parameters", description = "Gets current parameter values for a specific plugin. Uses batch reading for speed, falls back to feedback-based method if needed.")]
    async fn get_plugin_parameters_tool(
        &self,
        #[schemars(description = "Arguments for getting plugin parameters. Requires 'rid' and 'plugin_slot'.")]
        #[tool(aggr)] args: GetPluginParametersArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Requesting parameters for strip {}, plugin slot {}", args.rid, args.plugin_slot);

        let piid = args.plugin_slot; // 1-indexed for OSC
        let slot0 = args.plugin_slot - 1; // 0-indexed for state
        let mut params = Vec::new();

        // Try batch read first for faster and more reliable parameter reading
        let batch_success = if let Ok(_) = self.request_plugin_parameters_batch(args.rid, piid).await {
            if let Ok(batch_values) = self.await_batch_param_values(args.rid, slot0, 1200).await {
                // Batch read successful - map values to parameter IDs
                // The batch response contains all control parameters in order (by parameter index: 0, 1, 2, ...)
                // We need to match them with all control parameters (both input and output)
                
                // Build mapping: for each parameter index, find the corresponding parameter ID if it's a control parameter
                let mut param_index_to_id: Vec<Option<i32>> = Vec::new();
                {
                    let state = self.ardour_state.lock().await;
                    // Iterate through possible parameter indices (0-based)
                    // Most plugins have < 200 parameters
                    for param_index in 0..200 {
                        let param_id = param_index + 1; // Convert to 1-based parameter ID
                        // Check if this parameter ID exists in parameter_mapping and is a control parameter
                        let is_control = state.parameter_mapping
                            .iter()
                            .any(|((ssid, sl, pid), (is_input, is_output, _))| {
                                *ssid == args.rid && *sl == slot0 && *pid == param_id && (*is_input || *is_output)
                            });
                        if is_control {
                            param_index_to_id.push(Some(param_id));
                        } else {
                            param_index_to_id.push(None);
                        }
                    }
                }
                
                // Now match batch values with control parameters
                // batch_values[i] corresponds to the i-th control parameter
                let mut control_param_count = 0;
                
                for (param_index, param_id_opt) in param_index_to_id.iter().enumerate() {
                    if let Some(param_id) = param_id_opt {
                        // This is a control parameter
                        if control_param_count < batch_values.len() {
                            let value = batch_values[control_param_count];
                            
                            // Get parameter name from cache
                            let pname = {
                                let state = self.ardour_state.lock().await;
                                state.plugin_parameter_names.get(&(args.rid, slot0, *param_id)).cloned().unwrap_or_else(|| format!("param_{}", param_id))
                            };
                            
                            params.push(json!({
                                "id": param_id,
                                "name": pname,
                                "value": value
                            }));
                            
                            control_param_count += 1;
                        }
                    }
                }
                
                // Verify we got all control parameters
                if control_param_count == batch_values.len() {
                    tracing::info!("Batch read successful: got {} parameters for strip {} slot {}", params.len(), args.rid, args.plugin_slot);
                    true
                } else {
                    tracing::warn!("Batch read: got {} control params from {} batch values, falling back to feedback method", 
                        control_param_count, batch_values.len());
                    params.clear();
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        
        // Fallback to feedback-based method if batch failed
        if !batch_success {
            tracing::info!("Using feedback-based parameter reading for strip {} slot {}", args.rid, args.plugin_slot);
            
            // Check if plugin is already selected to avoid unnecessary selection overhead
            let needs_selection = {
                let state = self.ardour_state.lock().await;
                state.selected_strip != Some(args.rid) || state.selected_plugin_slot != Some(args.plugin_slot)
            };

            if needs_selection {
                // First select the plugin (this will trigger feedback)
                let select_result = self.select_plugin_tool(SelectPluginArgs {
                    rid: args.rid,
                    plugin_slot: args.plugin_slot,
                }).await;

                if let Ok(CallToolResult { is_error: Some(true), .. }) = select_result {
                    return select_result; // Return error if selection failed
                }

                // Wait longer for all parameter feedback to arrive after selection
                // Parameters arrive asynchronously, so we need to wait for them
                tokio::time::sleep(Duration::from_millis(1200)).await;
            } else {
                // Plugin already selected - just wait briefly for any pending parameter updates
                tokio::time::sleep(Duration::from_millis(300)).await;
            }

            // Read back the parameters from state
            let state = self.ardour_state.lock().await;

            // Look for parameters matching this strip and slot
            // Note: The slot might be stored as the visible index, so we need to check
            // both the requested slot and the currently selected slot
            let target_slot = args.plugin_slot;
            let current_selected_slot = state.selected_plugin_slot;
            
            tracing::info!("Retrieving parameters for strip {}, slot {} (currently selected slot: {:?})", 
                          args.rid, target_slot, current_selected_slot);
            
            // Debug: log all stored parameters for this strip
            let all_params_for_strip: Vec<_> = state.plugin_parameters.iter()
                .filter(|((ssid, _, _), _)| *ssid == args.rid)
                .collect();
            tracing::info!("Found {} stored parameters for strip {}: {:?}", 
                          all_params_for_strip.len(), args.rid,
                          all_params_for_strip.iter().map(|((_, slot, param_id), _)| format!("slot={}, param={}", slot, param_id)).collect::<Vec<_>>());

            for ((ssid, slot, param_id), value) in state.plugin_parameters.iter() {
                // Match by strip ID and either the requested slot or the currently selected slot
                // This handles the case where /select/plugin uses visible index but we query by actual slot
                if *ssid == args.rid && (*slot == target_slot || (current_selected_slot.is_some() && *slot == current_selected_slot.unwrap())) {
                    let name = state.plugin_parameter_names
                        .get(&(*ssid, *slot, *param_id))
                        .or_else(|| state.plugin_parameter_names.get(&(args.rid, target_slot, *param_id)))
                        .cloned()
                        .unwrap_or_else(|| format!("param_{}", param_id));

                    params.push(json!({
                        "id": param_id,
                        "name": name,
                        "value": value
                    }));
                    
                    tracing::info!("Found parameter: strip={}, stored_slot={}, param={}, name={}, value={}", 
                                  ssid, slot, param_id, name, value);
                }
            }
        }

        // Sort by param_id for consistent output
        params.sort_by_key(|p| p["id"].as_i64().unwrap_or(0));

        let result_json = json!({
            "strip_id": args.rid,
            "plugin_slot": args.plugin_slot,
            "parameter_count": params.len(),
            "parameters": params
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result_json)
                .unwrap_or_else(|_| "Failed to serialize parameters".to_string())
        )]))
    }

    #[tool(name = "get_meter_readings", description = "Gets meter readings from meters.lv2 plugins on a track. Looks for TPnRMSstereo, EBUr128, stereoscope, and spectr30stereo plugins and reads their control output ports.")]
    async fn get_meter_readings_tool(
        &self,
        #[schemars(description = "Arguments for getting meter readings. Requires 'rid' (integer).")]
        #[tool(aggr)] args: GetMeterReadingsArgs
    ) -> Result<CallToolResult, McpError> {
        // #region agent log
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"METER_RUST_ENTRY","location":"main.rs:1464","message":"get_meter_readings_tool: ENTRY","data":{{"rid":{}}},"timestamp":{}}}"#, args.rid, timestamp);
        }
        // #endregion
        tracing::info!("Getting meter readings for strip {}", args.rid);

        if args.rid <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                format!("Invalid rid: {}. Must be a positive integer.", args.rid)
            )]));
        }

        // Meter plugin names to look for (from meters.lv2)
        let meter_plugin_names = vec![
            "True-Peak and RMS Meter (Stereo)",
            "True-Peak and RMS Meter",
            "EBU R128 Meter",
            "Stereo/Frequency Scope",
            "1/3 Octave Spectrum Display Stereo",
        ];

        let mut meter_readings = json!({
            "strip_id": args.rid,
            "meters_found": [],
            "combined_data": {}
        });

        // Get plugin list from state directly (don't parse Content)
        // Trigger plugin list update by calling list_strip_plugins (only once)
        let _ = self.list_strip_plugins_tool(ListStripPluginsArgs { rid: args.rid }).await;

        // Use the cached state (no sleeps; we rely on existing plugin list cache)
        let plugin_list_snapshot = {
            let state = self.ardour_state.lock().await;
            state.plugin_lists.get(&args.rid).cloned()
        };

        if let Some(plugin_list) = plugin_list_snapshot {
            // #region agent log
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                let plugin_count = plugin_list.len();
                let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"METER_RUST_1","location":"main.rs:1493","message":"get_meter_readings: Got plugin list","data":{{"rid":{},"plugin_count":{}}},"timestamp":{}}}"#, args.rid, plugin_count, timestamp);
            }
            // #endregion
            let mut found_meters = Vec::new();
            
            for (slot, name, enabled) in &plugin_list {
                // Check if this is a meter plugin
                let is_meter = meter_plugin_names.iter().any(|&meter_name| {
                    name.contains(meter_name) || meter_name.contains(name)
                });
                
                // #region agent log
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                    let name_escaped = name.replace('"', "\\\"");
                    let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"METER_RUST_2","location":"main.rs:1496","message":"get_meter_readings: Checking plugin","data":{{"rid":{},"slot":{},"name":"{}","enabled":{},"is_meter":{}}},"timestamp":{}}}"#, args.rid, slot, name_escaped, enabled, is_meter, timestamp);
                }
                // #endregion
                
                if is_meter && *enabled {
                    tracing::info!("Found meter plugin on strip {}: {} (slot {})", args.rid, name, slot);

                    // Request plugin descriptor (so we learn output ports + labels)
                    let piid = *slot + 1; // 1-indexed for OSC
                    let _ = self.request_plugin_descriptor(args.rid, piid).await;
                    let _ = self.await_descriptor_done(args.rid, *slot, 1500).await;

                    // Collect output control ports for this plugin, then query their current values
                    let output_params: Vec<i32> = {
                        let state = self.ardour_state.lock().await;
                        state.parameter_mapping
                            .iter()
                            .filter_map(|((ssid, sl, pid), (_is_input, is_output, _))| {
                                if *ssid == args.rid && *sl == *slot && *is_output {
                                    Some(*pid)
                                } else {
                                    None
                                }
                            })
                            .collect()
                    };

                    let mut params_json = Vec::new();
                    
                    // Try batch read first for faster performance
                    let batch_success = if let Ok(_) = self.request_plugin_parameters_batch(args.rid, piid).await {
                        if let Ok(batch_values) = self.await_batch_param_values(args.rid, *slot, 1200).await {
                            // Batch read successful - map values to parameter IDs
                            // The batch response contains all control parameters in order (by parameter index: 0, 1, 2, ...)
                            // We need to match them with output_params
                            // Strategy: Build a mapping of parameter index -> parameter ID for control parameters
                            // by iterating through parameter indices and checking parameter_mapping
                            
                            // Build mapping: for each parameter index, find the corresponding parameter ID if it's a control parameter
                            let mut param_index_to_id: Vec<Option<i32>> = Vec::new();
                            {
                                let state = self.ardour_state.lock().await;
                                // Iterate through possible parameter indices (0-based)
                                // Most plugins have < 200 parameters
                                for param_index in 0..200 {
                                    let param_id = param_index + 1; // Convert to 1-based parameter ID
                                    // Check if this parameter ID exists in parameter_mapping and is a control parameter
                                    let is_control = state.parameter_mapping
                                        .iter()
                                        .any(|((ssid, sl, pid), (is_input, is_output, _))| {
                                            *ssid == args.rid && *sl == *slot && *pid == param_id && (*is_input || *is_output)
                                        });
                                    if is_control {
                                        param_index_to_id.push(Some(param_id));
                                    } else {
                                        param_index_to_id.push(None);
                                    }
                                }
                            }
                            
                            // Now match batch values with output parameters
                            // batch_values[i] corresponds to the i-th control parameter
                            // We need to find which parameter ID that is
                            let output_params_set: std::collections::HashSet<i32> = output_params.iter().cloned().collect();
                            let mut control_param_count = 0;
                            
                            for (param_index, param_id_opt) in param_index_to_id.iter().enumerate() {
                                if let Some(param_id) = param_id_opt {
                                    // This is a control parameter
                                    if control_param_count < batch_values.len() {
                                        let value = batch_values[control_param_count];
                                        
                                        // Check if this is an output parameter we want
                                        if output_params_set.contains(&param_id) {
                                            let pname = {
                                                let state = self.ardour_state.lock().await;
                                                state.plugin_parameter_names.get(&(args.rid, *slot, *param_id)).cloned().unwrap_or_else(|| format!("param_{}", param_id))
                                            };
                                            params_json.push(json!({
                                                "id": param_id,
                                                "name": pname,
                                                "value": value
                                            }));
                                        }
                                        
                                        control_param_count += 1;
                                    }
                                }
                            }
                            
                            // Verify we got all output parameters
                            if params_json.len() == output_params.len() && control_param_count == batch_values.len() {
                                true
                            } else {
                                tracing::warn!("Batch read: got {} output params (expected {}) from {} batch values ({} control params), falling back to individual reads", 
                                    params_json.len(), output_params.len(), batch_values.len(), control_param_count);
                                params_json.clear();
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    
                    // Fallback to individual reads if batch failed or didn't match
                    if !batch_success {
                        for pid in &output_params {
                            let _ = self.request_plugin_parameter_value(args.rid, piid, *pid).await;
                            if let Ok(v) = self.await_param_value(args.rid, *slot, *pid, 800).await {
                                let pname = {
                                    let state = self.ardour_state.lock().await;
                                    state.plugin_parameter_names.get(&(args.rid, *slot, *pid)).cloned().unwrap_or_else(|| format!("param_{}", pid))
                                };
                                params_json.push(json!({
                                    "id": pid,
                                    "name": pname,
                                    "value": v
                                }));
                            }
                        }
                    }

                    found_meters.push(json!({
                        "plugin_name": name,
                        "slot": piid,
                        "enabled": enabled,
                        "parameters": params_json
                    }));
                }
            }
            
            meter_readings["meters_found"] = json!(found_meters);
            
            // Combine meter data into unified format
            let mut combined = json!({});
            for meter in &found_meters {
                if let Some(params) = meter.get("parameters").and_then(|p| p.as_array()) {
                    for param in params {
                        if let (Some(name), Some(value)) = (
                            param.get("name").and_then(|n| n.as_str()),
                            param.get("value").and_then(|v| v.as_f64())
                        ) {
                            combined[name] = json!(value);
                        }
                    }
                }
            }
            // Extract keys before moving combined
            let combined_keys: Vec<String> = combined.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
            let found_count = found_meters.len();
            meter_readings["combined_data"] = combined;
            // #region agent log
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"METER_RUST_6","location":"main.rs:1547","message":"get_meter_readings: Final result","data":{{"rid":{},"meters_found_count":{},"combined_data_keys":{:?}}},"timestamp":{}}}"#, args.rid, found_count, combined_keys, timestamp);
            }
            // #endregion
        } else {
            tracing::warn!("No plugin list found for strip {}", args.rid);
            // #region agent log
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"METER_RUST_7","location":"main.rs:1549","message":"get_meter_readings: No plugin list found","data":{{"rid":{}}},"timestamp":{}}}"#, args.rid, timestamp);
            }
            // #endregion
        }

        // Request transport_frame from Ardour to ensure we have the latest value
        // Ardour only sends /transport_frame as a reply when requested, not automatically
        if let Err(e) = self.send_osc_message("/transport_frame", None).await {
            tracing::warn!("Failed to request transport_frame from Ardour: {}. Using cached value if available.", e);
        } else {
            // Wait a bit for Ardour to send the reply
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        
        // Add transport_frame timestamp to meter readings
        let transport_frame = {
            let state = self.ardour_state.lock().await;
            state.transport_frame
        };
        if let Some(frame) = transport_frame {
            meter_readings["transport_frame"] = json!(frame);
        } else {
            meter_readings["transport_frame"] = json!(null);
        }

        let result_json_str = serde_json::to_string_pretty(&meter_readings)
            .unwrap_or_else(|_| "Failed to serialize meter readings".to_string());
        
        // #region agent log
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            let result_preview = if result_json_str.len() > 200 { &result_json_str[..200] } else { &result_json_str };
            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"METER_RUST_EXIT","location":"main.rs:1607","message":"get_meter_readings_tool: RETURNING RESULT","data":{{"rid":{},"result_preview":"{}"}},"timestamp":{}}}"#, args.rid, result_preview.replace('"', "\\\""), timestamp);
        }
        // #endregion
        
        Ok(CallToolResult::success(vec![Content::text(result_json_str)]))
    }

    #[tool(name = "get_session_sample_rate", description = "Gets the current Ardour session sample rate in Hz. Returns integer sample rate (e.g., 44100, 48000, 96000).")]
    async fn get_session_sample_rate_tool(
        &self,
        #[schemars(description = "No arguments required.")]
        #[tool(aggr)] _args: GetSessionSampleRateArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Getting session sample rate from Ardour via native OSC");

        let osc_args = vec![osc::Type::Int(9099)]; // reply port
        if let Err(e) = self.send_osc_message("/mixpilot/get_sample_rate", Some(osc_args)).await {
            tracing::warn!("Failed to send get_sample_rate OSC: {}. Defaulting to 48000 Hz", e);
            let result = json!({ "sample_rate": 48000 });
            let result_str = serde_json::to_string_pretty(&result)
                .unwrap_or_else(|_| "{\"sample_rate\":48000}".to_string());
            return Ok(CallToolResult::success(vec![Content::text(result_str)]));
        }

        match self.await_sample_rate(1000).await {
            Ok(sr) => {
                tracing::info!("Received sample rate: {} Hz", sr);
                let result = json!({ "sample_rate": sr });
                let result_str = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| format!("{{\"sample_rate\":{}}}", sr));
                Ok(CallToolResult::success(vec![Content::text(result_str)]))
            }
            Err(e) => {
                tracing::warn!("Timed out or failed getting sample rate: {}. Defaulting to 48000 Hz", e);
                let result = json!({ "sample_rate": 48000 });
                let result_str = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| "{\"sample_rate\":48000}".to_string());
                Ok(CallToolResult::success(vec![Content::text(result_str)]))
            }
        }
    }

    #[tool(name = "get_loop_range", description = "Gets the current loop range from Ardour session. Returns loop start and end in samples and seconds, or null if no loop is set.")]
    async fn get_loop_range_tool(
        &self,
        #[schemars(description = "No arguments required.")]
        #[tool(aggr)] _args: GetLoopRangeArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Getting loop range from Ardour via native OSC");

        let osc_args = vec![osc::Type::Int(9099)]; // reply port
        if let Err(e) = self.send_osc_message("/mixpilot/get_loop_range", Some(osc_args)).await {
            tracing::warn!("Failed to send get_loop_range OSC: {}. Returning null", e);
            return Ok(CallToolResult::success(vec![Content::text("null".to_string())]));
        }

        match self.await_loop_range(1000).await {
            Ok(Some((start, end, start_sec, end_sec))) => {
                let result = json!({
                    "loop_start_samples": start,
                    "loop_end_samples": end,
                    "loop_start_seconds": start_sec,
                    "loop_end_seconds": end_sec
                });
                let result_str = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| "null".to_string());
                tracing::info!("Loop range: {} - {} samples", start, end);
                Ok(CallToolResult::success(vec![Content::text(result_str)]))
            }
            Ok(None) => {
                tracing::info!("No loop range set");
                Ok(CallToolResult::success(vec![Content::text("null".to_string())]))
            }
            Err(e) => {
                tracing::warn!("Failed to get loop range: {}. Returning null", e);
                Ok(CallToolResult::success(vec![Content::text("null".to_string())]))
            }
        }
    }

    #[tool(name = "set_region_parameter", description = "Sets a plugin parameter value for a specific time range (region). Uses region FX to apply the change only within the specified time range. First adds the plugin to regions if needed, then sets the parameter.")]
    async fn set_region_parameter_tool(
        &self,
        #[schemars(description = "Arguments for setting region parameter. Requires route, plugin_name, param_name, value, start_samples, end_samples. plugin_uri is optional but recommended.")]
        #[tool(aggr)] args: SetRegionParameterArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Setting region parameter: route={}, plugin={}, param={}, value={}, range={}-{}",
            args.route, args.plugin_name, args.param_name, args.value, args.start_samples, args.end_samples
        );
        
        // Get route ID - try parsing as integer first, otherwise look up by name
        let route_id: i32 = match args.route.parse() {
            Ok(id) => id,
            Err(_) => {
                // TODO: Look up route ID from name using strip list
                // For now, return error if not numeric
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Route name lookup not yet implemented. Please use numeric route ID. Got: '{}'",
                    args.route
                ))]));
            }
        };
        
        // Get plugin URI
        let plugin_uri = match args.plugin_uri {
            Some(uri) => uri,
            None => {
                match get_plugin_uri_from_name(&args.plugin_name) {
                    Some(uri) => {
                        tracing::info!("Resolved plugin URI from name '{}': {}", args.plugin_name, uri);
                        uri
                    }
                    None => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Plugin URI not provided and could not resolve from plugin name '{}'. Please provide plugin_uri.",
                            args.plugin_name
                        ))]));
                    }
                }
            }
        };
        
        // Convert samples to int32 (OSC limitation - max ~24 hours at 48kHz)
        let start_samples_i32 = args.start_samples.min(i32::MAX as i64).max(i32::MIN as i64) as i32;
        let end_samples_i32 = args.end_samples.min(i32::MAX as i64).max(i32::MIN as i64) as i32;
        
        if start_samples_i32 as i64 != args.start_samples || end_samples_i32 as i64 != args.end_samples {
            tracing::warn!("Sample range truncated to int32: {} -> {}, {} -> {}", 
                args.start_samples, start_samples_i32, args.end_samples, end_samples_i32);
        }
        
        // Step 1: Add Region FX plugin to regions at time range
        tracing::info!("Adding Region FX plugin '{}' to regions at range {}-{}", plugin_uri, start_samples_i32, end_samples_i32);
        let add_plugin_args = vec![
            osc::Type::Int(route_id),
            osc::Type::Int(start_samples_i32),
            osc::Type::Int(end_samples_i32),
            osc::Type::String(plugin_uri.clone()),
        ];
        
        match self.send_osc_message("/strip/region/add_plugin", Some(add_plugin_args)).await {
            Ok(_) => {
                tracing::info!("Successfully sent /strip/region/add_plugin command");
            }
            Err(e) => {
                tracing::warn!("Failed to add Region FX plugin (may already exist): {}", e);
                // Continue anyway - plugin might already exist
            }
        }
        
        // Small delay to allow Ardour to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Step 2: Set parameter on Region FX plugins
        tracing::info!("Setting parameter '{}' = {} on Region FX '{}'", args.param_name, args.value, args.plugin_name);
        let set_param_args = vec![
            osc::Type::Int(route_id),
            osc::Type::Int(start_samples_i32),
            osc::Type::Int(end_samples_i32),
            osc::Type::String(args.plugin_name.clone()),
            osc::Type::String(args.param_name.clone()),
            osc::Type::Float(args.value),
        ];
        
        match self.send_osc_message("/strip/region/plugin/parameter", Some(set_param_args)).await {
            Ok(_) => {
                tracing::info!("Successfully sent /strip/region/plugin/parameter command");
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Regional parameter set: {} on {} ({}={}) for range {}-{} samples",
                    args.plugin_name, args.route, args.param_name, args.value, args.start_samples, args.end_samples
                ))]))
            }
            Err(e) => {
                let error_msg = format!(
                    "Failed to set Region FX parameter: {}. Plugin may need to be added first or parameter name may be incorrect.",
                    e
                );
                tracing::error!("{}", error_msg);
                Ok(CallToolResult::error(vec![Content::text(error_msg)]))
            }
        }
    }

    #[tool(name = "write_plugin_automation", description = "Writes an automation point for a plugin parameter at a specific time position. Used for bypassing track plugins during regional edits.")]
    async fn write_plugin_automation_tool(
        &self,
        #[schemars(description = "Arguments for writing plugin automation. Requires route_id, plugin_slot, param_id, time_samples, value.")]
        #[tool(aggr)] args: WritePluginAutomationArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Writing plugin automation via native OSC: route_id={}, plugin_slot={}, param_id={}, time_samples={}, value={}",
            args.route_id, args.plugin_slot, args.param_id, args.time_samples, args.value
        );

        // Convert 0-indexed slot/param to 1-indexed for Ardour OSC
        let piid_1idx = args.plugin_slot + 1;
        let par_1idx = args.param_id + 1;

        let osc_args = vec![
            osc::Type::Int(args.route_id),
            osc::Type::Int(piid_1idx),
            osc::Type::Int(par_1idx),
            osc::Type::Int(args.time_samples as i32),
            osc::Type::Float(args.value),
        ];

        match self.send_osc_message("/mixpilot/write_plugin_auto", Some(osc_args)).await {
            Ok(_) => {
                let result = json!({
                    "success": true,
                    "route_id": args.route_id,
                    "plugin_slot": args.plugin_slot,
                    "param_id": args.param_id,
                    "time_samples": args.time_samples,
                    "value": args.value
                });
                let result_str = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| "{\"success\":true}".to_string());
                tracing::info!("Successfully sent plugin automation point");
                Ok(CallToolResult::success(vec![Content::text(result_str)]))
            }
            Err(e) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to write plugin automation: {}", e
                ))]))
            }
        }
    }

    #[tool(name = "read_automation_value", description = "Reads the current automation value for a plugin parameter or track control (gain, pan) at a specific time position. Returns the effective value (from automation or manual setting).")]
    async fn read_automation_value_tool(
        &self,
        #[schemars(description = "Arguments for reading automation value. Requires route_id, control_type ('plugin', 'gain', 'pan_azimuth', or 'pan_width'), time_samples. For 'plugin' control_type, also requires plugin_slot and param_id.")]
        #[tool(aggr)] args: ReadAutomationValueArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Reading automation value via native OSC: route_id={}, control_type={}, time_samples={}",
            args.route_id, args.control_type, args.time_samples
        );

        // Validate control_type
        if args.control_type != "plugin" && args.control_type != "gain" &&
           args.control_type != "pan_azimuth" && args.control_type != "pan_width" {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid control_type: {}. Must be 'plugin', 'gain', 'pan_azimuth', or 'pan_width'",
                args.control_type
            ))]));
        }

        if args.control_type == "plugin" {
            if args.plugin_slot.is_none() || args.param_id.is_none() {
                return Ok(CallToolResult::error(vec![Content::text(
                    "plugin_slot and param_id are required for 'plugin' control_type".to_string()
                )]));
            }
            let piid_1idx = args.plugin_slot.unwrap() + 1;
            let par_1idx = args.param_id.unwrap() + 1;

            let osc_args = vec![
                osc::Type::Int(args.route_id),
                osc::Type::Int(piid_1idx),
                osc::Type::Int(par_1idx),
                osc::Type::Int(args.time_samples as i32),
                osc::Type::Int(9099), // reply port
            ];

            if let Err(e) = self.send_osc_message("/mixpilot/read_plugin_auto", Some(osc_args)).await {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to send read_plugin_auto: {}", e
                ))]));
            }
        } else {
            let osc_args = vec![
                osc::Type::Int(args.route_id),
                osc::Type::String(args.control_type.clone()),
                osc::Type::Int(args.time_samples as i32),
                osc::Type::Int(9099), // reply port
            ];

            if let Err(e) = self.send_osc_message("/mixpilot/read_track_auto", Some(osc_args)).await {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to send read_track_auto: {}", e
                ))]));
            }
        }

        match self.await_auto_value(1500).await {
            Ok(value) => {
                let result = json!({
                    "success": true,
                    "value": value,
                    "route_id": args.route_id,
                    "control_type": args.control_type,
                    "time_samples": args.time_samples
                });
                let result_str = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| format!("{{\"success\":true,\"value\":{}}}", value));
                tracing::info!("Read automation value: {}", value);
                Ok(CallToolResult::success(vec![Content::text(result_str)]))
            }
            Err(e) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to read automation value: {}", e
                ))]))
            }
        }
    }

    #[tool(name = "write_track_automation", description = "Writes an automation point for a track-level control (gain, pan_azimuth, pan_width) at a specific time position.")]
    async fn write_track_automation_tool(
        &self,
        #[schemars(description = "Arguments for writing track automation. Requires route_id, control_type ('gain', 'pan_azimuth', or 'pan_width'), time_samples, value. use_current_as_start is optional.")]
        #[tool(aggr)] args: WriteTrackAutomationArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Writing track automation via native OSC: route_id={}, control_type={}, time_samples={}, value={}, use_current_as_start={:?}",
            args.route_id, args.control_type, args.time_samples, args.value, args.use_current_as_start
        );

        // Validate control_type
        if args.control_type != "gain" && args.control_type != "pan_azimuth" && args.control_type != "pan_width" {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Invalid control_type: {}. Must be 'gain', 'pan_azimuth', or 'pan_width'",
                args.control_type
            ))]));
        }

        let osc_args = vec![
            osc::Type::Int(args.route_id),
            osc::Type::String(args.control_type.clone()),
            osc::Type::Int(args.time_samples as i32),
            osc::Type::Float(args.value),
        ];

        match self.send_osc_message("/mixpilot/write_track_auto", Some(osc_args)).await {
            Ok(_) => {
                let result = json!({
                    "success": true,
                    "route_id": args.route_id,
                    "control_type": args.control_type,
                    "time_samples": args.time_samples,
                    "value": args.value
                });
                let result_str = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| "{\"success\":true}".to_string());
                tracing::info!("Successfully sent track automation point");
                Ok(CallToolResult::success(vec![Content::text(result_str)]))
            }
            Err(e) => {
                Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to write track automation: {}", e
                ))]))
            }
        }
    }

    #[tool(name = "list_strip_plugins", description = "Lists all plugins on a specific strip. First requests plugin list from Ardour, then returns cached or known data.")]
    async fn list_strip_plugins_tool(
        &self,
        #[schemars(description = "Arguments for listing plugins. Requires 'rid' (integer).")]
        #[tool(aggr)] args: ListStripPluginsArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Listing plugins for strip {}", args.rid);

        if args.rid <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                format!("Invalid rid: {}. Must be a positive integer.", args.rid)
            )]));
        }

        // Request plugin list from Ardour via OSC
        tracing::info!("Requesting plugin list from Ardour for strip {}", args.rid);
        match self.send_osc_message("/strip/plugin/list", Some(vec![osc::Type::Int(args.rid)])).await {
            Ok(_) => {
                tracing::info!("Sent /strip/plugin/list request for strip {}", args.rid);
            }
            Err(e) => {
                tracing::warn!("Failed to send /strip/plugin/list request: {}", e);
            }
        }

        // Wait for Ardour to send the reply, with retries
        // Check not just that the key exists, but that the list is non-empty
        let mut plugin_list_received = false;
        for _attempt in 0..5 {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            let state_check = self.ardour_state.lock().await;
            if let Some(cached_list) = state_check.plugin_lists.get(&args.rid) {
                // Only consider it received if the list is non-empty
                if !cached_list.is_empty() {
                    plugin_list_received = true;
                    drop(state_check);
                    break;
                }
            }
            drop(state_check);
        }
        
        if !plugin_list_received {
            tracing::warn!("Plugin list for strip {} not received (or empty) after waiting, using cache if available", args.rid);
        }

        let state = self.ardour_state.lock().await;

        // First, check if we have a cached plugin list from Ardour
        let mut plugins = Vec::new();
        if let Some(plugin_list) = state.plugin_lists.get(&args.rid) {
            tracing::info!("Using cached plugin list for strip {}: {} plugins", args.rid, plugin_list.len());
            // Use the actual plugin list from Ardour
            for (slot, name, enabled) in plugin_list {
                let param_count = state.plugin_parameters.keys()
                    .filter(|(s, sl, _)| *s == args.rid && *sl == *slot)
                    .count();
                
                plugins.push(json!({
                    "slot": slot + 1,  // Convert 0-indexed to 1-indexed for API
                    "name": name,
                    "enabled": enabled,
                    "known_parameters": param_count,
                    "has_parameter_names": state.plugin_parameter_names.keys()
                        .any(|(s, sl, _)| *s == args.rid && *sl == *slot)
                }));
            }
        } else {
            // Fallback: Extract unique plugin slots we've seen from parameter data
            let mut plugin_slots = std::collections::HashSet::new();
            for (ssid, slot, _) in state.plugin_parameters.keys() {
                if *ssid == args.rid {
                    plugin_slots.insert(*slot);
                }
            }

            plugins = plugin_slots.iter().map(|slot| {
                let param_count = state.plugin_parameters.keys()
                    .filter(|(s, sl, _)| *s == args.rid && *sl == *slot)
                    .count();
                
                json!({
                    "slot": slot,
                    "known_parameters": param_count,
                    "has_parameter_names": state.plugin_parameter_names.keys()
                        .any(|(s, sl, _)| *s == args.rid && *sl == *slot)
                })
            }).collect();
        }

        let result_json = json!({
            "strip_id": args.rid,
            "plugin_count": plugins.len(),
            "plugins": plugins
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&result_json)
                .unwrap_or_else(|_| "Failed to serialize plugin list".to_string())
        )]))
    }

    #[tool(name = "request_strip_list", description = "Requests the complete list of all strips (tracks and buses) from Ardour. This triggers Ardour to send /strip/list feedback with all strips.")]
    async fn request_strip_list_tool(
        &self,
        #[schemars(description = "No arguments required. Requests Ardour to send the complete strip list.")]
        #[tool(aggr)] _args: RequestStripListArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!("Requesting strip list from Ardour");
        
        // Send /strip/list request to Ardour
        // According to Ardour OSC docs, /strip/list with no arguments requests the list
        match self.send_osc_message("/strip/list", None).await {
            Ok(_) => {
                tracing::info!("Sent /strip/list request to Ardour");
                
                // Wait a bit for Ardour to respond
                tokio::time::sleep(Duration::from_millis(500)).await;
                
                // Check how many strips we have now
                let state = self.ardour_state.lock().await;
                let strip_count = state.strip_list.iter().filter(|ti| ti.id != 0).count();
                drop(state);
                
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Strip list requested from Ardour. Currently have {} strips in cache.",
                    strip_count
                ))]))
            }
            Err(e) => {
                let error_msg = format!("Failed to send /strip/list request: {}", e);
                tracing::error!("{}", error_msg);
                Ok(CallToolResult::error(vec![Content::text(error_msg)]))
            }
        }
    }

    #[tool(name = "set_strip_pan_stereo_width", description = "Sets the stereo width for a panner on a stereo strip.")]
    async fn set_strip_pan_stereo_width_tool(
        &self,
        #[schemars(description = "Arguments for setting stereo panner width. Requires 'rid' and 'width' (0.0-1.0).")]
        #[tool(aggr)] args: SetStripPanStereoWidthArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_pan_stereo_width_tool for rid: {}, width: {}",
            args.rid, args.width
        );

        if !(0.0..=1.0).contains(&args.width) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid width: must be between 0.0 and 1.0 (inclusive).".to_string()
            )]));
        }

        let osc_args = vec![
            osc::Type::Int(args.rid),
            osc::Type::Float(args.width),
        ];
        let address = "/strip/panner/width";

        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} stereo pan width set to {}.",
                args.rid, args.width
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_selected_strip_pan_stereo_width", description = "Sets the stereo width for the currently selected stereo strip's panner.")]
    async fn set_selected_strip_pan_stereo_width_tool(
        &self,
        #[schemars(description = "Arguments for setting selected strip stereo panner width. Requires 'width' (0.0-1.0).")]
        #[tool(aggr)] args: SetSelectedStripPanStereoWidthArgs
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_selected_strip_pan_stereo_width_tool with width: {}",
            args.width
        );

        if !(0.0..=1.0).contains(&args.width) {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid width: must be between 0.0 and 1.0 (inclusive).".to_string()
            )]));
        }

        let osc_args = vec![
            osc::Type::Float(args.width),
        ];
        let address = "/select/pan_stereo_width";

        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Selected strip stereo pan width set to {}.
 (Ensure a strip was selected prior to calling this)",
                args.width
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_strip_send_gain_db", description = "Sets a strip send gain in dB via /strip/send/gain (rid, send_id, gain_db).")]
    async fn set_strip_send_gain_db_tool(
        &self,
        #[schemars(description = "Args: rid (int), send_id (1-based int), gain_db (float dB).")]
        #[tool(aggr)] args: SetStripSendGainDbArgs,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_send_gain_db_tool rid={}, send_id={}, gain_db={}",
            args.rid,
            args.send_id,
            args.gain_db
        );

        if args.rid <= 0 || args.send_id <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid rid/send_id: must be positive integers.".to_string(),
            )]));
        }

        let address = "/strip/send/gain";
        let osc_args = vec![
            osc::Type::Int(args.rid),
            osc::Type::Int(args.send_id),
            osc::Type::Float(args.gain_db),
        ];

        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} send {} gain set to {} dB.",
                args.rid, args.send_id, args.gain_db
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_strip_send_fader", description = "Sets a strip send fader via /strip/send/fader (rid, send_id, fader).")]
    async fn set_strip_send_fader_tool(
        &self,
        #[schemars(description = "Args: rid (int), send_id (1-based int), fader (float).")]
        #[tool(aggr)] args: SetStripSendFaderArgs,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_send_fader_tool rid={}, send_id={}, fader={}",
            args.rid,
            args.send_id,
            args.fader
        );

        if args.rid <= 0 || args.send_id <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid rid/send_id: must be positive integers.".to_string(),
            )]));
        }

        let address = "/strip/send/fader";
        let osc_args = vec![
            osc::Type::Int(args.rid),
            osc::Type::Int(args.send_id),
            osc::Type::Float(args.fader),
        ];

        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} send {} fader set to {}.",
                args.rid, args.send_id, args.fader
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_strip_send_enable", description = "Enables/disables a strip send via /strip/send/enable (rid, send_id, enable_state).")]
    async fn set_strip_send_enable_tool(
        &self,
        #[schemars(description = "Args: rid (int), send_id (1-based int), enable_state (bool).")]
        #[tool(aggr)] args: SetStripSendEnableArgs,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_send_enable_tool rid={}, send_id={}, enable_state={}",
            args.rid,
            args.send_id,
            args.enable_state
        );

        if args.rid <= 0 || args.send_id <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid rid/send_id: must be positive integers.".to_string(),
            )]));
        }

        let address = "/strip/send/enable";
        let enable_val = if args.enable_state { 1.0f32 } else { 0.0f32 };
        let osc_args = vec![
            osc::Type::Int(args.rid),
            osc::Type::Int(args.send_id),
            osc::Type::Float(enable_val),
        ];

        match self.send_osc_message(address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} send {} enable set to {}.",
                args.rid, args.send_id, args.enable_state
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }

    #[tool(name = "set_strip_polarity", description = "Sets strip polarity invert via /strip/<rid>/polarity (value 0/1).")]
    async fn set_strip_polarity_tool(
        &self,
        #[schemars(description = "Args: rid (int), invert_state (bool).")]
        #[tool(aggr)] args: SetStripPolarityArgs,
    ) -> Result<CallToolResult, McpError> {
        tracing::info!(
            "Executing set_strip_polarity_tool rid={}, invert_state={}",
            args.rid,
            args.invert_state
        );

        if args.rid <= 0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid rid: must be a positive integer.".to_string(),
            )]));
        }

        // Ardour's OSC strip handler supports polarity under /strip/<rid>/polarity
        let address = format!("/strip/{}/polarity", args.rid);
        let yn = if args.invert_state { 1i32 } else { 0i32 };
        let osc_args = vec![osc::Type::Int(yn)];

        match self.send_osc_message(&address, Some(osc_args)).await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Strip {} polarity invert set to {}.",
                args.rid, args.invert_state
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "OSC send error for {}: {}",
                address, e
            ))])),
        }
    }
}

impl ServerHandler for ArdourService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            server_info: Implementation {
                name: "ardour-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: Some(false),
                }),
                resources: Some(ResourcesCapability {
                    subscribe: Some(false),
                    list_changed: Some(false),
                }),
                prompts: None,
                experimental: None,
                logging: None,
            },
            instructions: Some("Ardour MCP server for OSC control.".to_string()),
        }
    }

    async fn list_tools(
        &self,
        _request: PaginatedRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, McpError> {
        Ok(rmcp::model::ListToolsResult {
            next_cursor: None,
            tools: ArdourService::tool_box().list(),
        })
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParam,
        ctx: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CallToolResult, McpError> {
        // #region agent log
        let tool_name = request.name.clone();
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            let args_preview = format!("{:?}", request.arguments).replace('"', "\\\"");
            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"MCP_ROUTER_ENTRY","location":"main.rs:2033","message":"call_tool: Routing tool call","data":{{"tool_name":"{}","args_preview":"{}"}},"timestamp":{}}}"#, tool_name, args_preview, timestamp);
        }
        // #endregion
        let tool_call_context = rmcp::handler::server::tool::ToolCallContext::new(self, request, ctx);
        let result = ArdourService::tool_box().call(tool_call_context).await;
        // #region agent log
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
            let is_ok = result.is_ok();
            let is_error = result.as_ref().ok().and_then(|r| r.is_error).unwrap_or(false);
            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"MCP_ROUTER_EXIT","location":"main.rs:2039","message":"call_tool: Tool call completed","data":{{"tool_name":"{}","is_ok":{},"is_error":{}}},"timestamp":{}}}"#, tool_name, is_ok, is_error, timestamp);
        }
        // #endregion
        result
    }

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let mut raw_playback_state_resource = RawResource::new(
            "ardour:/state/playback",
            "Ardour Playback State"
        );
        raw_playback_state_resource.description = Some(
            "Current playback state of Ardour (e.g., Playing, Stopped, Unknown).".to_string()
        );
        let playback_state_resource: Resource = raw_playback_state_resource.no_annotation();

        let mut raw_transport_frame_resource = RawResource::new(
            "ardour:/state/transport_frame",
            "Ardour Transport Frame Position"
        );
        raw_transport_frame_resource.description = Some(
            "Current playhead position in samples. Returns 'Unknown' if not yet reported by Ardour.".to_string()
        );
        let transport_frame_resource: Resource = raw_transport_frame_resource.no_annotation();
        
        let mut raw_plugin_params_resource = RawResource::new(
            "ardour:/plugin/parameters",
            "Plugin Parameter Values"
        );
        raw_plugin_params_resource.description = Some(
            "Current parameter values for all plugins in the session.".to_string()
        );
        let plugin_params_resource: Resource = raw_plugin_params_resource.no_annotation();
        
        let all_resources = vec![playback_state_resource, transport_frame_resource, plugin_params_resource];

        tracing::debug!("Listing resources. Count: {}, Content: {:?}", all_resources.len(), all_resources);
        
        Ok(ListResourcesResult { 
            resources: all_resources,
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let resource_uri = request.uri.as_str();
        match resource_uri {
            "ardour:/state/playback" => {
                let state = self.ardour_state.lock().await;
                let status_str = match state.playback_status {
                    PlaybackStatus::Playing => "Playing",
                    PlaybackStatus::Stopped => "Stopped",
                    PlaybackStatus::Unknown => "Unknown",
                };
                Ok(ReadResourceResult {
                    contents: vec![rmcp::model::ResourceContents::TextResourceContents {
                        uri: resource_uri.to_string(),
                        mime_type: Some("text/plain".to_string()),
                        text: status_str.to_string(),
                    }],
                })
            }
            "ardour:/strip/list" => {
                let state = self.ardour_state.lock().await;
                let valid_strips: Vec<&TrackInfo> = state.strip_list.iter().filter(|ti| ti.id != 0).collect();
                
                match serde_json::to_string_pretty(&valid_strips) {
                    Ok(json_response) => Ok(ReadResourceResult {
                        contents: vec![rmcp::model::ResourceContents::TextResourceContents {
                            uri: resource_uri.to_string(),
                            mime_type: Some("application/json".to_string()),
                            text: json_response,
                        }],
                    }),
                    Err(e) => {
                        tracing::error!("Failed to serialize strip list: {}", e);
                        Err(McpError::internal_error(
                            format!("Failed to serialize strip list: {}", e),
                            None
                        ))
                    }
                }
            }
            "ardour:/state/transport_frame" => {
                let state = self.ardour_state.lock().await;
                let frame_str = match state.transport_frame {
                    Some(frame) => frame.to_string(),
                    None => "Unknown".to_string(),
                };
                Ok(ReadResourceResult {
                    contents: vec![rmcp::model::ResourceContents::TextResourceContents {
                        uri: resource_uri.to_string(),
                        mime_type: Some("text/plain".to_string()),
                        text: frame_str,
                    }],
                })
            }
            "ardour:/action/list" => {
                let placeholder_actions = json!([
                    { "name": "Session/Save", "description": "Saves the current session." },
                    { "name": "Editor/zoom-to-session", "description": "Zooms to fit the entire session." },
                    { "name": "Transport/Loop", "description": "Toggles loop playback." }
                ]);
                Ok(ReadResourceResult {
                    contents: vec![rmcp::model::ResourceContents::TextResourceContents {
                        uri: resource_uri.to_string(),
                        mime_type: Some("application/json".to_string()),
                        text: placeholder_actions.to_string(),
                    }],
                })
            }
            "ardour:/plugin/parameters" => {
                let state = self.ardour_state.lock().await;
                
                // Build JSON structure: { "strips": { ssid: { "plugins": { slot: { "parameters": {...} } } } } } }
                let mut strips_map = serde_json::Map::new();
                
                // Group parameters by strip and slot
                let mut strip_plugins: HashMap<i32, HashMap<i32, serde_json::Map<String, serde_json::Value>>> = HashMap::new();
                
                for ((ssid, slot, param_id), value) in state.plugin_parameters.iter() {
                    let param_name = state.plugin_parameter_names
                        .get(&(*ssid, *slot, *param_id))
                        .cloned()
                        .unwrap_or_else(|| format!("param_{}", param_id));
                    
                    let strip_entry = strip_plugins.entry(*ssid).or_insert_with(HashMap::new);
                    let plugin_entry = strip_entry.entry(*slot).or_insert_with(serde_json::Map::new);
                    
                    let mut param_obj = serde_json::Map::new();
                    param_obj.insert("id".to_string(), json!(*param_id));
                    param_obj.insert("name".to_string(), json!(param_name));
                    param_obj.insert("value".to_string(), json!(*value));
                    
                    plugin_entry.insert(format!("param_{}", param_id), json!(param_obj));
                }
                
                // Convert to JSON structure
                for (ssid, plugins) in strip_plugins {
                    let mut plugins_map = serde_json::Map::new();
                    for (slot, params) in plugins {
                        let mut plugin_obj = serde_json::Map::new();
                        plugin_obj.insert("slot".to_string(), json!(slot));
                        plugin_obj.insert("parameters".to_string(), json!(params));
                        plugins_map.insert(slot.to_string(), json!(plugin_obj));
                    }
                    
                    let mut strip_obj = serde_json::Map::new();
                    strip_obj.insert("id".to_string(), json!(ssid));
                    strip_obj.insert("plugins".to_string(), json!(plugins_map));
                    strips_map.insert(ssid.to_string(), json!(strip_obj));
                }
                
                let mut result = serde_json::Map::new();
                result.insert("strips".to_string(), json!(strips_map));
                
                match serde_json::to_string_pretty(&result) {
                    Ok(json_text) => Ok(ReadResourceResult {
                        contents: vec![rmcp::model::ResourceContents::TextResourceContents {
                            uri: resource_uri.to_string(),
                            mime_type: Some("application/json".to_string()),
                            text: json_text,
                        }],
                    }),
                    Err(e) => {
                        tracing::error!("Failed to serialize plugin parameters: {}", e);
                        Err(McpError::internal_error(
                            format!("Failed to serialize plugin parameters: {}", e),
                            None
                        ))
                    }
                }
            }
            _ => {
                Err(McpError::resource_not_found(
                    format!("Resource URI '{}' not found.", resource_uri),
                    Some(json!({ "uri": resource_uri }))
                ))
            }
        }
    }

    async fn list_prompts(
        &self,
        _request: PaginatedRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult { 
            prompts: vec![],
            next_cursor: None,
        })
    }

    async fn get_prompt(
        &self,
        _req: GetPromptRequestParam,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        Err(McpError::invalid_params("Prompt not found", None))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create log directory if it doesn't exist
    let log_dir = Path::new("logs");
    if !log_dir.exists() {
        std::fs::create_dir_all(log_dir)?;
    }
    let log_file_path = log_dir.join("ardour_mcp_server.log");

    // Create or append to the log file
    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(log_file_path)?;

    // Create a combined writer for file and stderr
    let stderr_writer = std::io::stderr.with_max_level(tracing::Level::INFO); // Log INFO and above to stderr
    let file_writer = log_file.with_max_level(tracing::Level::DEBUG); // Log DEBUG and above to file
    let combined_writer = stderr_writer.and(file_writer);

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::DEBUG.into()))
        .with_writer(combined_writer) // Use the combined writer
        .with_ansi(true) // Enable ANSI for terminal, will be ignored by file
        .init();

    tracing::info!("\n======================================================================\nNEW SERVER RUN: {}\n======================================================================", chrono::Local::now().to_rfc2822());

    let ardour_service = ArdourService::new()?;
    
    // Send initial OSC setup to Ardour
    if let Err(e) = ardour_service.send_osc_setup_to_ardour().await {
        tracing::warn!("Could not send initial OSC setup to Ardour: {}. Feedback might not work.", e);
        // Decide if this should be a fatal error or just a warning. For now, warning.
    }

    let ardour_state_clone = Arc::clone(&ardour_service.ardour_state);
    let pending_clone = Arc::clone(&ardour_service.pending_requests);
    
    let server_process = ardour_service.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("MCP Server serving error: {:?}", e);
    })?;

    tracing::info!("Ardour MCP server started and waiting for connections...");

    let _osc_listener_handle = tokio::spawn(async move {
        if let Err(e) = listen_ardour_osc_events(ardour_state_clone, pending_clone).await {
            tracing::error!("OSC listener task failed: {:?}", e);
        }
    });

    server_process.waiting().await?; 

    tracing::info!("Ardour MCP server stopped.");

    Ok(())
}

async fn listen_ardour_osc_events(state: Arc<Mutex<ArdourState>>, pending: Arc<Mutex<PendingRequests>>) -> Result<()> { 
    tracing::info!(
        "Starting OSC listener for Ardour events on {}",
        MCP_SERVER_OSC_LISTEN_ADDR
    );

    let listen_socket = tokio::net::UdpSocket::bind(MCP_SERVER_OSC_LISTEN_ADDR)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind Tokio UDP socket for OSC on {}: {}", MCP_SERVER_OSC_LISTEN_ADDR, e))?;
    
    tracing::info!("Tokio UDP socket for OSC bound to {}", MCP_SERVER_OSC_LISTEN_ADDR);
    
    let mut buf = [0u8; osc::recv::DEFAULT_MTU];

    loop {
        match listen_socket.recv_from(&mut buf).await {
            Ok((size, peer_addr)) => {
                let packet = osc::decode(&buf[..size])
                    .map_err(|e| anyhow::anyhow!("OSC decode error from {}: {}", peer_addr, e))?;
                handle_osc_packet(packet, peer_addr, Arc::clone(&state), Arc::clone(&pending)).await;
            }
            Err(e) => {
                // Log non-fatal errors, but break on others
                tracing::error!("OSC recv_from error: {}. Listener might stop.", e);
                // Consider if we should break or continue on certain errors.
                // For now, let's break on any error to avoid tight loops on persistent issues.
                break Err(anyhow::anyhow!("OSC recv_from error: {}", e)); 
            }
        }
    }
}

async fn handle_osc_packet(packet: osc::Packet, peer_addr: std::net::SocketAddr, state: Arc<Mutex<ArdourState>>, pending: Arc<Mutex<PendingRequests>>) {
    match packet {
        osc::Packet::Message(msg) => {
            // #region agent log - Log ALL OSC messages to debug transport_state issue
            let is_transport_related = msg.addr == "/transport_state" || msg.addr == "/transport_frame" || msg.addr == "/strip/play";
            if is_transport_related {
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                    let args_preview = format!("{:?}", msg.args);
                    let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"TRANSPORT_OSC_ALL","location":"main.rs:3246","message":"OSC_DEBUG: Received transport-related OSC message","data":{{"addr":"{}","args_preview":"{}"}},"timestamp":{}}}"#, msg.addr, args_preview.replace('"', "\\\""), timestamp);
                }
            }
            // #endregion
            
            // Check playback status before logging - only log when playback is active
            let should_log = {
                let state_guard = state.lock().await;
                matches!(state_guard.playback_status, PlaybackStatus::Playing)
            };
            
            // Log OSC messages to help debug plugin parameter feedback
            // Only log plugin-related messages, not meter/signal feedback
            // Only log when playback is active (Playing), suppress when Stopped or Unknown
            if msg.addr.contains("plugin") || 
               (msg.addr.contains("select") && (msg.addr.contains("plugin") || msg.addr == "/select/plugin")) {
                if should_log {
                    tracing::info!("Received OSC message from {}: {} {:?}", peer_addr, msg.addr, msg.args);
                }
            } else {
                if should_log {
                    tracing::debug!("Received OSC message from {}: {} {:?}", peer_addr, msg.addr, msg.args);
                }
            }
            
            // Handle /strip/name/<ssid> <name>
            if msg.addr.starts_with("/strip/name/") {
                let parts: Vec<&str> = msg.addr.split('/').collect();
                if parts.len() == 4 { // expecting "", "strip", "name", "<ssid>"
                    if let Ok(ssid) = parts[3].parse::<i32>() {
                        if ssid > 0 { // SSIDs are 1-indexed
                            if let Some(osc::Type::String(name)) = msg.args.get(0) {
                                tracing::info!("Ardour feedback: /strip/name/{} -> {}", ssid, name);
            let mut current_state = state.lock().await;
                                let vec_idx = (ssid - 1) as usize;

                                // Ensure strip_list is large enough
                                if vec_idx >= current_state.strip_list.len() {
                                    current_state.strip_list.resize_with(vec_idx + 1, || TrackInfo {
                                        id: 0, // Will be overwritten by actual ssid if this is the target new strip
                                        name: String::new(),
                                        track_type: "unknown".to_string(),
                                    });
                                }
                                
                                // Update or fill the slot
                                let strip_info = &mut current_state.strip_list[vec_idx];
                                strip_info.id = ssid; // Set/confirm the ID
                                strip_info.name = name.clone();
                                // track_type will be updated if/when we get a /strip/type message
                                tracing::debug!("Updated strip_list: SSID {}, Name '{}', Type '{}'", strip_info.id, strip_info.name, strip_info.track_type);
                            }
                        }
                    }
                }
            } 
            // Handle /strip/type/<ssid> <type_str_or_int> (Hypothetical, based on common patterns)
            else if msg.addr.starts_with("/strip/type/") { // Note the 'else if'
                let parts: Vec<&str> = msg.addr.split('/').collect();
                if parts.len() == 4 { // expecting "", "strip", "type", "<ssid>"
                    if let Ok(ssid) = parts[3].parse::<i32>() {
                        if ssid > 0 { 
                            if let Some(type_arg) = msg.args.get(0) {
                                let type_str = match type_arg {
                                    osc::Type::String(s) => s.clone(),
                                    osc::Type::Int(i) => format!("type_id_{}", i), // Or map to known string types
                                    _ => "unknown_type_format".to_string(),
                                };
                                tracing::info!("Ardour feedback: /strip/type/{} -> {}", ssid, type_str);
                                let mut current_state = state.lock().await;
                                let vec_idx = (ssid - 1) as usize;

                                if vec_idx < current_state.strip_list.len() {
                                    // Only update if strip already known from a /name message (or pre-sized)
                                    let strip_info = &mut current_state.strip_list[vec_idx];
                                    if strip_info.id == ssid { // Ensure it's the correct strip, not a placeholder from resize
                                        strip_info.track_type = type_str;
                                        tracing::debug!("Updated strip_list: SSID {}, Name '{}', Type '{}'", strip_info.id, strip_info.name, strip_info.track_type);
                                    } else {
                                        tracing::warn!("Received /strip/type/{} but strip_list[{}] has id {}, expected {}. Type not updated.", ssid, vec_idx, strip_info.id, ssid);
                                    }
                                } else {
                                    tracing::warn!("Received /strip/type/{} but strip {} is out of bounds for current strip_list (len {}). Type not updated.", ssid, ssid, current_state.strip_list.len());
                                    // Optionally, could create a new entry here if /type can come before /name
                                    // current_state.strip_list.resize_with(vec_idx + 1, || TrackInfo { id: 0, name: String::new(), track_type: "unknown".to_string() });
                                    // current_state.strip_list[vec_idx].id = ssid;
                                    // current_state.strip_list[vec_idx].track_type = type_str;
                                }
                            }
                        }
                    }
                }
            }
            // Handle /transport_state <state_int> <speed_float> (no /ardour/ prefix, new arg order)
            else if msg.addr == "/transport_state" { 
                // #region agent log
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                    let args_preview = format!("{:?}", msg.args);
                    let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"TRANSPORT_STATE_RECEIVED","location":"main.rs:3335","message":"OSC_DEBUG: Received /transport_state message","data":{{"args_count":{},"args_preview":"{}"}},"timestamp":{}}}"#, msg.args.len(), args_preview.replace('"', "\\\""), timestamp);
                }
                // #endregion
                
                if msg.args.len() == 2 { // Expecting state and speed
                    let transport_state_val = msg.args.get(0).and_then(|arg| if let osc::Type::Int(s) = arg { Some(*s) } else { None });
                    let speed = msg.args.get(1).and_then(|arg| if let osc::Type::Float(s) = arg { Some(*s) } else { None });

                    if let (Some(ts_val), Some(s)) = (transport_state_val, speed) {
                        tracing::info!("Ardour feedback: /transport_state state: {}, speed: {}", ts_val, s);
                        
                        // #region agent log
                        let old_status = {
                            let state_guard = state.lock().await;
                            format!("{:?}", state_guard.playback_status)
                        };
                        // #endregion
                        
                        let mut current_state_guard = state.lock().await;
                        let new_status = match ts_val {
                            0 => { // Stopped
                                current_state_guard.playback_status = PlaybackStatus::Stopped;
                                "Stopped"
                            }
                            1 => { // Rolling (Playing)
                                current_state_guard.playback_status = PlaybackStatus::Playing;
                                "Playing"
                            }
                            2 => { // Looping (also Playing)
                                current_state_guard.playback_status = PlaybackStatus::Playing;
                                "Playing"
                            }
                            _ => {
                                tracing::warn!("Received /transport_state with unknown state value: {}", ts_val);
                                "Unknown"
                            }
                        };
                        
                        // #region agent log
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"TRANSPORT_STATE_UPDATED","location":"main.rs:3343","message":"OSC_DEBUG: Playback status updated from /transport_state","data":{{"transport_state_val":{},"old_status":"{}","new_status":"{}","speed":{}}},"timestamp":{}}}"#, ts_val, old_status, new_status, s, timestamp);
                        }
                        // #endregion
                        
                        tracing::info!("Playback status updated to {} via /transport_state", new_status);
                    } else {
                        tracing::warn!("Received /transport_state with unexpected argument types: {:?}. Expected Int, Float.", msg.args);
                    }
                } else {
                    tracing::warn!("Received /transport_state with incorrect number of arguments: {}. Expected 2.", msg.args.len());
                }
            }
            // Handle /transport_frame <frame_int64> (no /ardour/ prefix)
            else if msg.addr == "/transport_frame" { // Changed from "/ardour/transport_frame"
                if msg.args.len() == 1 {
                    if let Some(osc::Type::Long(frame)) = msg.args.get(0) {
                        tracing::info!("Ardour feedback: /transport_frame -> {}", frame);
                        let mut current_state_guard = state.lock().await;
                        current_state_guard.transport_frame = Some(*frame);
                    } else {
                        tracing::warn!("Received /transport_frame with unexpected argument type: {:?}. Expected Long.", msg.args.get(0));
                    }
                } else {
                    tracing::warn!("Received /transport_frame with incorrect number of arguments: {}. Expected 1.", msg.args.len());
                }
            }
            // Handle playback status updates from /strip/play (original logic)
            else if msg.addr.as_str() == "/strip/play" { // Note the 'else if'
                // #region agent log
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                    let args_preview = format!("{:?}", msg.args);
                    let old_status = {
                        let state_guard = state.lock().await;
                        format!("{:?}", state_guard.playback_status)
                    };
                    let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"STRIP_PLAY_RECEIVED","location":"main.rs:3419","message":"OSC_DEBUG: Received /strip/play message","data":{{"args_preview":"{}","old_status":"{}"}},"timestamp":{}}}"#, args_preview.replace('"', "\\\""), old_status, timestamp);
                }
                // #endregion
                
                let mut current_state = state.lock().await; // Moved lock inside specific message handling
                    if let Some(osc::Type::Int(is_playing_val)) = msg.args.get(0) {
                        let new_status = if *is_playing_val == 1 {
                            current_state.playback_status = PlaybackStatus::Playing;
                            "Playing"
                        } else if *is_playing_val == 0 {
                            current_state.playback_status = PlaybackStatus::Stopped;
                            "Stopped"
                        } else {
                            "Unknown"
                        };
                        
                        // #region agent log
                        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("/Users/mariasaldana/mixpilot/.cursor/debug.log") {
                            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                            let _ = writeln!(file, r#"{{"sessionId":"debug-session","runId":"run1","hypothesisId":"STRIP_PLAY_UPDATED","location":"main.rs:3422","message":"OSC_DEBUG: Playback status updated from /strip/play","data":{{"is_playing_val":{},"new_status":"{}"}},"timestamp":{}}}"#, is_playing_val, new_status, timestamp);
                        }
                        // #endregion
                        
                        if *is_playing_val == 1 {
                            tracing::info!("Ardour feedback via /strip/play: Playback Started (state=1)");
                        } else if *is_playing_val == 0 {
                            tracing::info!("Ardour feedback via /strip/play: Playback Stopped (state=0)");
                        } else {
                            tracing::warn!(
                                "Ardour feedback via /strip/play: Received with unexpected integer state: {}. Ignoring for playback status.",
                                is_playing_val
                            );
                        }
                    } else {
                        tracing::warn!(
                            "Ardour feedback via /strip/play: Received without expected integer argument. Args: {:?}. Ignoring for playback status.",
                            msg.args
                        );
                    }
                }
            // NEW: Handle plugin parameter values for selected plugin
            // Ardour sends /select/plugin/parameter/<param_id> with value as argument
            // Exclude /select/plugin/parameter/name/* messages (those are handled separately)
            else if msg.addr.starts_with("/select/plugin/parameter/") && !msg.addr.starts_with("/select/plugin/parameter/name") {
                // Extract param_id from path like /select/plugin/parameter/1
                let param_id = if let Some(param_id_str) = msg.addr.strip_prefix("/select/plugin/parameter/") {
                    match param_id_str.parse::<i32>() {
                        Ok(id) => id,
                        Err(_) => {
                            tracing::warn!("Received /select/plugin/parameter/<id> with invalid param_id in path: {}", param_id_str);
                            return;
                        }
                    }
                } else {
                    tracing::warn!("Received /select/plugin/parameter message but couldn't extract param_id from path: {}", msg.addr);
                    return;
                };
                
                // Value is the first (and only) argument
                let value = match msg.args.get(0) {
                    Some(osc::Type::Float(v)) => {
                        if !v.is_finite() {
                            tracing::warn!("Received /select/plugin/parameter/{} with non-finite value: {}", param_id, v);
                            return;
                        }
                        *v
                    }
                    Some(osc::Type::Int(i)) => *i as f32,  // Some plugins might send Int
                    Some(other) => {
                        tracing::warn!("Received /select/plugin/parameter/{} with invalid value type: {:?}. Expected Float or Int.", param_id, other);
                        return;
                    }
                    None => {
                        tracing::error!("Received /select/plugin/parameter/{} with missing value argument", param_id);
                        return;
                    }
                };
                
                let mut current_state = state.lock().await;
                if let (Some(ssid), Some(slot)) = (current_state.selected_strip, current_state.selected_plugin_slot) {
                    let key = (ssid, slot, param_id);
                    current_state.plugin_parameters.insert(key, value);
                    tracing::info!("Plugin parameter feedback: strip={}, slot={}, param={}, value={}", 
                                  ssid, slot, param_id, value);
                } else {
                    // Try to use the most recently requested slot if available
                    // This handles cases where parameters arrive before selection is confirmed
                    if let Some(ssid) = current_state.selected_strip {
                        if let Some(slot) = current_state.selected_plugin_slot {
                            let key = (ssid, slot, param_id);
                            current_state.plugin_parameters.insert(key, value);
                            tracing::info!("Plugin parameter feedback (using cached slot): strip={}, slot={}, param={}, value={}", 
                                          ssid, slot, param_id, value);
                        } else {
                            tracing::warn!("Received /select/plugin/parameter but no plugin slot set (strip={:?}, plugin={:?})", 
                                          current_state.selected_strip, current_state.selected_plugin_slot);
                        }
                    } else {
                        tracing::warn!("Received /select/plugin/parameter but no strip/plugin selected (strip={:?}, plugin={:?})", 
                                      current_state.selected_strip, current_state.selected_plugin_slot);
                    }
                }
            }
            // NEW: Handle plugin parameter names
            // Handle plugin parameter names
            // Ardour sends /select/plugin/parameter/name/<param_id> with name as argument
            else if msg.addr.starts_with("/select/plugin/parameter/name/") {
                // Extract param_id from path like /select/plugin/parameter/name/1
                let param_id = if let Some(param_id_str) = msg.addr.strip_prefix("/select/plugin/parameter/name/") {
                    match param_id_str.parse::<i32>() {
                        Ok(id) => id,
                        Err(_) => {
                            tracing::warn!("Received /select/plugin/parameter/name/<id> with invalid param_id in path: {}", param_id_str);
                            return;
                        }
                    }
                } else {
                    tracing::warn!("Received /select/plugin/parameter/name message but couldn't extract param_id from path: {}", msg.addr);
                    return;
                };
                
                // Name is the first (and only) argument
                let name = match msg.args.get(0) {
                    Some(osc::Type::String(n)) => {
                        if n.is_empty() {
                            tracing::debug!("Received /select/plugin/parameter/name/{} with empty name", param_id);
                            return;
                        }
                        n.clone()
                    }
                    Some(other) => {
                        tracing::warn!("Received /select/plugin/parameter/name/{} with invalid name type: {:?}. Expected String.", param_id, other);
                        return;
                    }
                    None => {
                        tracing::error!("Received /select/plugin/parameter/name/{} with missing name argument", param_id);
                        return;
                    }
                };
                
                let mut current_state = state.lock().await;
                if let (Some(ssid), Some(slot)) = (current_state.selected_strip, current_state.selected_plugin_slot) {
                    let key = (ssid, slot, param_id);
                    current_state.plugin_parameter_names.insert(key, name.clone());
                    tracing::info!("Plugin parameter name: strip={}, slot={}, param={}, name={}", 
                                  ssid, slot, param_id, name);
                } else {
                    tracing::warn!("Received /select/plugin/parameter/name/{} but no strip/plugin selected (strip={:?}, plugin={:?})", 
                                  param_id, current_state.selected_strip, current_state.selected_plugin_slot);
                }
            }
            // NEW: Handle plugin selection feedback (when Ardour confirms a plugin is selected)
            else if msg.addr == "/select/plugin" || msg.addr.starts_with("/select/plugin/") {
                // Try to extract plugin slot from message or args
                // Format might be /select/plugin/<slot> or /select/plugin with slot in args
                let plugin_slot = if msg.addr.contains('/') && msg.addr != "/select/plugin" {
                    // Try to parse slot from path like /select/plugin/1
                    let parts: Vec<&str> = msg.addr.split('/').collect();
                    if parts.len() >= 4 {
                        parts[3].parse::<i32>().ok()
                    } else {
                        None
                    }
                } else if !msg.args.is_empty() {
                    // Try to get slot from first argument
                    match msg.args.get(0) {
                        Some(osc::Type::Int(slot)) => Some(*slot),
                        Some(osc::Type::Float(f)) => Some(*f as i32),
                        _ => None,
                    }
                } else {
                    None
                };

                if let Some(slot) = plugin_slot {
                    let mut current_state = state.lock().await;
                    if let Some(ssid) = current_state.selected_strip {
                        current_state.selected_plugin_slot = Some(slot);
                        tracing::info!("Plugin selection confirmed: strip={}, plugin_slot={}", ssid, slot);
                    } else {
                        tracing::debug!("Received plugin selection but no strip selected yet");
                    }
                } else {
                    tracing::debug!("Received /select/plugin message but couldn't extract plugin slot: addr={}, args={:?}", msg.addr, msg.args);
                }
            }
            // Handle /strip/plugin/list reply from Ardour
            // Format: /strip/plugin/list <ssid> <piid+1> <name> <enabled> [<piid+1> <name> <enabled> ...]
            // piid is 0-indexed slot number, Ardour sends piid+1 (1-indexed)
            else if msg.addr == "/strip/plugin/list" {
                if msg.args.len() < 1 {
                    tracing::warn!("Received /strip/plugin/list with insufficient arguments");
                    return;
                }

                // First argument is ssid
                let ssid = match msg.args.get(0) {
                    Some(osc::Type::Int(id)) => *id,
                    Some(osc::Type::Float(f)) => *f as i32,
                    _ => {
                        tracing::warn!("Received /strip/plugin/list with invalid ssid");
                        return;
                    }
                };

                let mut plugin_list = Vec::new();
                let mut i = 1;
                // Parse plugin entries: each plugin has (piid+1, name, enabled)
                while i + 2 < msg.args.len() {
                    let piid_plus_one = match msg.args.get(i) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => break,
                    };
                    let slot = piid_plus_one - 1; // Convert back to 0-indexed

                    let name = match msg.args.get(i + 1) {
                        Some(osc::Type::String(n)) => n.clone(),
                        _ => break,
                    };

                    let enabled = match msg.args.get(i + 2) {
                        Some(osc::Type::Int(e)) => *e != 0,
                        Some(osc::Type::Float(f)) => *f != 0.0,
                        _ => break,
                    };

                    plugin_list.push((slot, name, enabled));
                    i += 3;
                }

                let mut current_state = state.lock().await;
                // Only store non-empty plugin lists to avoid overwriting valid data with empty lists
                // Empty lists can occur if parsing fails or if Ardour sends an incomplete response
                if !plugin_list.is_empty() {
                    // Check if we already have a plugin list for this strip
                    let existing_list_opt = current_state.plugin_lists.get(&ssid).cloned();
                    if let Some(existing_list) = existing_list_opt {
                        // Merge the new list with the existing one to avoid losing plugins
                        // This handles cases where Ardour sends incomplete lists
                        let mut merged_list = existing_list.clone();
                        let existing_slots: HashSet<i32> = existing_list.iter().map(|(slot, _, _)| *slot).collect();
                        let existing_len = existing_list.len();
                        
                        // Add plugins from new list that aren't in existing list
                        for (slot, name, enabled) in &plugin_list {
                            if !existing_slots.contains(slot) {
                                merged_list.push((*slot, name.clone(), *enabled));
                                tracing::info!("Merged plugin list: Added slot {} ({}) from new list", slot, name);
                            } else {
                                // Update existing entry if enabled state changed
                                if let Some(existing_entry) = merged_list.iter_mut().find(|(s, _, _)| *s == *slot) {
                                    existing_entry.2 = *enabled; // Update enabled state
                                    tracing::debug!("Updated enabled state for slot {}: {}", slot, enabled);
                                }
                            }
                        }
                        
                        // Sort by slot for consistency
                        merged_list.sort_by_key(|(slot, _, _)| *slot);
                        
                        current_state.plugin_lists.insert(ssid, merged_list.clone());
                        tracing::info!("Merged plugin list for strip {}: {} plugins total ({} from existing, {} new)", 
                                     ssid, merged_list.len(), existing_len, plugin_list.len());
                        for (slot, name, enabled) in &merged_list {
                            tracing::info!("  Slot {}: {} (enabled: {})", slot, name, enabled);
                        }
                    } else {
                        // No existing list - store the new one
                        current_state.plugin_lists.insert(ssid, plugin_list.clone());
                        tracing::info!("Received plugin list for strip {}: {} plugins", ssid, plugin_list.len());
                        for (slot, name, enabled) in &plugin_list {
                            tracing::info!("  Slot {}: {} (enabled: {})", slot, name, enabled);
                        }
                    }
                } else {
                    // Empty list received - check if we already have a non-empty list for this strip
                    let existing_list = current_state.plugin_lists.get(&ssid);
                    if let Some(existing) = existing_list {
                        if !existing.is_empty() {
                            // We have a valid non-empty list - don't overwrite with empty
                            tracing::warn!("Received empty plugin list for strip {} but cache already has {} plugins. Preserving existing cache. Args length: {}", 
                                         ssid, existing.len(), msg.args.len());
                            return; // Don't store the empty list
                        }
                    }
                    // No existing list or existing list is also empty - store it
                    tracing::warn!("Received empty plugin list for strip {} - storing (no existing cache or existing is also empty). Args length: {}", 
                                 ssid, msg.args.len());
                    current_state.plugin_lists.insert(ssid, plugin_list.clone());
                }
            }
            // Handle /strip/plugin/inserted feedback from Ardour
            // Format: /strip/plugin/inserted <ssid> <slot> <plugin_name> <plugin_uri> <enabled>
            // slot is 0-indexed
            else if msg.addr == "/strip/plugin/inserted" {
                if msg.args.len() < 5 {
                    tracing::warn!("Received /strip/plugin/inserted with insufficient arguments (expected 5, got {})", msg.args.len());
                    return;
                }

                // Parse arguments: ssid, slot, plugin_name, plugin_uri, enabled
                let ssid = match msg.args.get(0) {
                    Some(osc::Type::Int(id)) => *id,
                    Some(osc::Type::Float(f)) => *f as i32,
                    _ => {
                        tracing::warn!("Received /strip/plugin/inserted with invalid ssid");
                        return;
                    }
                };

                let slot = match msg.args.get(1) {
                    Some(osc::Type::Int(s)) => *s,
                    Some(osc::Type::Float(f)) => *f as i32,
                    _ => {
                        tracing::warn!("Received /strip/plugin/inserted with invalid slot");
                        return;
                    }
                };

                let plugin_name = match msg.args.get(2) {
                    Some(osc::Type::String(n)) => n.clone(),
                    _ => {
                        tracing::warn!("Received /strip/plugin/inserted with invalid plugin_name");
                        return;
                    }
                };

                let plugin_uri = match msg.args.get(3) {
                    Some(osc::Type::String(u)) => u.clone(),
                    _ => {
                        tracing::warn!("Received /strip/plugin/inserted with invalid plugin_uri");
                        return;
                    }
                };

                let enabled = match msg.args.get(4) {
                    Some(osc::Type::Int(e)) => *e != 0,
                    Some(osc::Type::Float(f)) => *f != 0.0,
                    _ => {
                        tracing::warn!("Received /strip/plugin/inserted with invalid enabled");
                        return;
                    }
                };

                // Update plugin_lists cache
                let mut current_state = state.lock().await;
                let existing_list_opt = current_state.plugin_lists.get(&ssid).cloned();
                
                if let Some(mut existing_list) = existing_list_opt {
                    // Check if plugin already exists at this slot (update existing entry)
                    let mut found = false;
                    for entry in existing_list.iter_mut() {
                        if entry.0 == slot {
                            // Update existing entry
                            entry.1 = plugin_name.clone();
                            entry.2 = enabled;
                            found = true;
                            tracing::info!("Updated plugin at slot {} on strip {}: {} (enabled: {})", slot, ssid, plugin_name, enabled);
                            break;
                        }
                    }
                    
                    if !found {
                        // Add new plugin entry
                        existing_list.push((slot, plugin_name.clone(), enabled));
                        existing_list.sort_by_key(|(s, _, _)| *s);
                        tracing::info!("Added plugin at slot {} on strip {}: {} (enabled: {})", slot, ssid, plugin_name, enabled);
                    }
                    
                    current_state.plugin_lists.insert(ssid, existing_list);
                } else {
                    // No existing list - create new one with this plugin
                    let new_list = vec![(slot, plugin_name.clone(), enabled)];
                    current_state.plugin_lists.insert(ssid, new_list);
                    tracing::info!("Created plugin list for strip {} with plugin at slot {}: {} (enabled: {})", ssid, slot, plugin_name, enabled);
                }
                
                tracing::info!("Plugin insertion feedback processed: strip={}, slot={}, name={}, uri={}, enabled={}", 
                             ssid, slot, plugin_name, plugin_uri, enabled);
            }
            // Handle /strip/plugin/descriptor responses
            // Format: /strip/plugin/descriptor i:ssid i:piid i:param_index s:label i:flags s:datatype f:lower f:upper s:print_fmt i:scale_points_count [scale_points...] d:current_value
            // Flags bit 0x80 = parameter_is_input
            // We use this to build a mapping of param_id -> visible_index for /select/plugin/parameter
            else if msg.addr == "/strip/plugin/descriptor" {
                if msg.args.len() >= 9 {
                    let ssid = match msg.args.get(0) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => {
                            tracing::warn!("Received /strip/plugin/descriptor with invalid ssid");
                            return;
                        }
                    };
                    let piid = match msg.args.get(1) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => {
                            tracing::warn!("Received /strip/plugin/descriptor with invalid piid");
                            return;
                        }
                    };
                    let param_index = match msg.args.get(2) {
                        Some(osc::Type::Int(idx)) => *idx,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => {
                            tracing::warn!("Received /strip/plugin/descriptor with invalid param_index");
                            return;
                        }
                    };
                    let label = match msg.args.get(3) {
                        Some(osc::Type::String(s)) => s.clone(),
                        _ => "unknown".to_string(),
                    };
                    let flags = match msg.args.get(4) {
                        Some(osc::Type::Int(f)) => *f,
                        _ => {
                            tracing::warn!("Received /strip/plugin/descriptor with invalid flags");
                            return;
                        }
                    };
                    let is_input = (flags & 0x80) != 0;
                    let is_output = (flags & 0x200) != 0;
                    
                    // Store in parameter mapping: (ssid, slot, param_id) -> (is_input, is_output, visible_index)
                    // For now, we'll calculate visible_index later when we have all parameters
                    let slot = piid - 1; // Convert to 0-indexed
                    let mut current_state = state.lock().await;
                    // We'll build the visible index mapping when we have all parameters
                    current_state.parameter_mapping.insert((ssid, slot, param_index), (is_input, is_output, None));
                    current_state.plugin_parameter_names.insert((ssid, slot, param_index), label);
                    // Best-effort capture current value (Ardour appends it as the last arg, usually Double)
                    if let Some(last) = msg.args.last() {
                        let v = match last {
                            osc::Type::Double(d) => Some(*d as f32),
                            osc::Type::Float(f) => Some(*f),
                            osc::Type::Int(i) => Some(*i as f32),
                            _ => None,
                        };
                        if let Some(v) = v {
                            current_state.plugin_parameters.insert((ssid, slot, param_index), v);
                        }
                    }
                    tracing::debug!("Received parameter descriptor: strip={}, slot={}, param={}, is_input={}, is_output={}", ssid, slot, param_index, is_input, is_output);
                }
            }
            // Handle /strip/plugin/descriptor_end - signals end of descriptor responses
            else if msg.addr == "/strip/plugin/descriptor_end" {
                if msg.args.len() >= 2 {
                    let ssid = match msg.args.get(0) {
                        Some(osc::Type::Int(id)) => *id,
                        _ => return,
                    };
                    let piid = match msg.args.get(1) {
                        Some(osc::Type::Int(id)) => *id,
                        _ => return,
                    };
                    let slot = piid - 1;
                    
                    // Now build the visible index mapping for this plugin
                    // Visible index = 1-indexed position in the list of INPUT parameters only
                    let mut current_state = state.lock().await;
                    let mut input_params: Vec<(i32, i32)> = Vec::new(); // (param_id, original_index)
                    
                    // Collect all input parameters for this plugin
                    for ((s, sl, param_id), (is_input, _, _)) in current_state.parameter_mapping.iter() {
                        if *s == ssid && *sl == slot && *is_input {
                            input_params.push((*param_id, *param_id));
                        }
                    }
                    
                    // Sort by param_id to get consistent ordering
                    input_params.sort_by_key(|(pid, _)| *pid);
                    
                    // Update visible_index for each input parameter
                    for (visible_idx, (param_id, _)) in input_params.iter().enumerate() {
                        let visible_index = (visible_idx + 1) as i32; // 1-indexed
                        if let Some(entry) = current_state.parameter_mapping.get_mut(&(ssid, slot, *param_id)) {
                            entry.2 = Some(visible_index);
                            tracing::debug!("Built visible index mapping: strip={}, slot={}, param={} -> visible_index={}", ssid, slot, param_id, visible_index);
                        }
                    }
                    
                    tracing::info!("Finished building parameter mapping for strip {}, plugin {}: {} input parameters mapped", ssid, piid, input_params.len());

                    drop(current_state);
                    let mut pending_state = pending.lock().await;
                    if let Some(tx) = pending_state.descriptor_done.remove(&(ssid, slot)) {
                        let _ = tx.send(());
                    }
                }
            }
            // NEW: /strip/plugin/parameter/value i:ssid i:piid i:param_index f:value
            else if msg.addr == "/strip/plugin/parameter/value" {
                if msg.args.len() >= 4 {
                    let ssid = match msg.args.get(0) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => return,
                    };
                    let piid = match msg.args.get(1) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => return,
                    };
                    let param_index = match msg.args.get(2) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => return,
                    };
                    let value = match msg.args.get(3) {
                        Some(osc::Type::Float(v)) => *v,
                        Some(osc::Type::Double(d)) => *d as f32,
                        Some(osc::Type::Int(i)) => *i as f32,
                        _ => return,
                    };
                    let slot = piid - 1;
                    {
                        let mut current_state = state.lock().await;
                        current_state.plugin_parameters.insert((ssid, slot, param_index), value);
                    }
                    let mut pending_state = pending.lock().await;
                    if let Some(tx) = pending_state.param_values.remove(&(ssid, slot, param_index)) {
                        let _ = tx.send(value);
                    }
                }
            }
            // NEW: /strip/plugin/parameter/values i:ssid i:piid i:param_count f:value1 f:value2 ...
            else if msg.addr == "/strip/plugin/parameter/values" {
                if msg.args.len() >= 3 {
                    let ssid = match msg.args.get(0) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => return,
                    };
                    let piid = match msg.args.get(1) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => return,
                    };
                    let _param_count = match msg.args.get(2) {
                        Some(osc::Type::Int(count)) => *count,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => return,
                    };
                    let slot = piid - 1;
                    
                    // Extract all parameter values (starting from index 3)
                    let mut values = Vec::new();
                    for i in 3..msg.args.len() {
                        if let Some(value) = match msg.args.get(i) {
                            Some(osc::Type::Float(v)) => Some(*v),
                            Some(osc::Type::Double(d)) => Some(*d as f32),
                            Some(osc::Type::Int(iv)) => Some(*iv as f32),
                            _ => None,
                        } {
                            values.push(value);
                        }
                    }
                    
                    // Store individual parameter values in state cache (for backward compatibility)
                    // Note: We don't have parameter indices in batch response, so we can't map them individually
                    // The batch response will be handled separately
                    
                    // Send batch values to waiting receiver
                    let mut pending_state = pending.lock().await;
                    if let Some(tx) = pending_state.batch_param_values.remove(&(ssid, slot)) {
                        let _ = tx.send(values);
                    }
                }
            }
            // MixPilot reply handlers for native OSC endpoints
            else if msg.addr == "/mixpilot/reply/sample_rate" {
                if let Some(osc::Type::Int(sr)) = msg.args.get(0) {
                    tracing::info!("Received /mixpilot/reply/sample_rate: {} Hz", sr);
                    let mut pending_state = pending.lock().await;
                    if let Some(tx) = pending_state.sample_rate.take() {
                        let _ = tx.send(*sr);
                    }
                }
            }
            else if msg.addr == "/mixpilot/reply/loop_range" {
                let mut pending_state = pending.lock().await;
                if let Some(tx) = pending_state.loop_range.take() {
                    if msg.args.len() == 1 {
                        // Single int 0 = no loop set
                        tracing::info!("Received /mixpilot/reply/loop_range: no loop");
                        let _ = tx.send(None);
                    } else if msg.args.len() >= 4 {
                        let start = match msg.args.get(0) {
                            Some(osc::Type::Int(v)) => *v,
                            _ => { let _ = tx.send(None); return; }
                        };
                        let end = match msg.args.get(1) {
                            Some(osc::Type::Int(v)) => *v,
                            _ => { let _ = tx.send(None); return; }
                        };
                        let start_sec = match msg.args.get(2) {
                            Some(osc::Type::Float(v)) => *v,
                            _ => { let _ = tx.send(None); return; }
                        };
                        let end_sec = match msg.args.get(3) {
                            Some(osc::Type::Float(v)) => *v,
                            _ => { let _ = tx.send(None); return; }
                        };
                        tracing::info!("Received /mixpilot/reply/loop_range: {} - {} samples", start, end);
                        let _ = tx.send(Some((start, end, start_sec, end_sec)));
                    } else {
                        let _ = tx.send(None);
                    }
                }
            }
            else if msg.addr == "/mixpilot/reply/auto_value" {
                if let Some(value) = match msg.args.get(0) {
                    Some(osc::Type::Float(v)) => Some(*v),
                    Some(osc::Type::Double(d)) => Some(*d as f32),
                    Some(osc::Type::Int(i)) => Some(*i as f32),
                    _ => None,
                } {
                    tracing::info!("Received /mixpilot/reply/auto_value: {}", value);
                    let mut pending_state = pending.lock().await;
                    if let Some(tx) = pending_state.auto_value.take() {
                        let _ = tx.send(value);
                    }
                }
            }
            // Handle /strip/list response from Ardour
            // Format: /strip/list <ssid1> <name1> <type1> [<ssid2> <name2> <type2> ...]
            else if msg.addr == "/strip/list" {
                tracing::info!("Received /strip/list response from Ardour with {} arguments", msg.args.len());
                
                let mut current_state = state.lock().await;
                
                // Clear the existing strip list when we receive a new /strip/list message
                // According to Ardour OSC docs, /strip/list with no args returns the complete list
                // Clearing ensures we don't accumulate stale entries from previous scans
                let previous_count = current_state.strip_list.iter().filter(|ti| ti.id != 0).count();
                current_state.strip_list.clear();
                
                let mut strips_added = 0;
                let mut strip_names_received = Vec::new();
                
                // Parse strip entries: each strip has (ssid, name, type)
                let mut i = 0;
                while i + 2 < msg.args.len() {
                    let ssid = match msg.args.get(i) {
                        Some(osc::Type::Int(id)) => *id,
                        Some(osc::Type::Float(f)) => *f as i32,
                        _ => {
                            tracing::warn!("Invalid ssid in /strip/list at position {}", i);
                            break;
                        }
                    };
                    
                    let name = match msg.args.get(i + 1) {
                        Some(osc::Type::String(n)) => n.clone(),
                        _ => {
                            tracing::warn!("Invalid name in /strip/list at position {}", i + 1);
                            break;
                        }
                    };
                    
                    let strip_type = match msg.args.get(i + 2) {
                        Some(osc::Type::String(t)) => t.clone(),
                        Some(osc::Type::Int(t)) => format!("type_{}", t),
                        _ => "unknown".to_string(),
                    };
                    
                    if ssid > 0 {
                        let vec_idx = (ssid - 1) as usize;
                        
                        // Ensure strip_list is large enough
                        if vec_idx >= current_state.strip_list.len() {
                            current_state.strip_list.resize_with(vec_idx + 1, || TrackInfo {
                                id: 0,
                                name: String::new(),
                                track_type: "unknown".to_string(),
                            });
                        }
                        
                        // Update or create the strip entry
                        let strip_info = &mut current_state.strip_list[vec_idx];
                        strip_info.id = ssid;
                        strip_info.name = name.clone();
                        strip_info.track_type = strip_type.clone();
                        
                        strips_added += 1;
                        strip_names_received.push(format!("{} (SSID {})", name, ssid));
                        tracing::debug!("Added/updated strip from /strip/list: SSID {}, Name '{}', Type '{}'", 
                                      ssid, name, strip_type);
                    }
                    
                    i += 3;
                }
                
                tracing::info!("Processed /strip/list response: {} strips received (was {} before clear). Strips: {}", 
                             strips_added, previous_count, strip_names_received.join(", "));
            }
            // NEW: Track strip selection (when a strip is selected, update selected_strip)
            else if msg.addr == "/strip/select" {
                match msg.args.get(0) {
                    Some(osc::Type::Int(ssid)) => {
                        if *ssid > 0 {
                            let mut current_state = state.lock().await;
                            current_state.selected_strip = Some(*ssid);
                            // Clear plugin selection when strip changes
                            current_state.selected_plugin_slot = None;
                            tracing::info!("Strip selected: {} (cleared plugin selection)", ssid);
                        } else {
                            tracing::warn!("Received /strip/select with invalid SSID: {}", ssid);
                        }
                    }
                    Some(other) => {
                        tracing::warn!("Received /strip/select with invalid argument type: {:?}. Expected Int.", other);
                    }
                    None => {
                        tracing::error!("Received /strip/select with missing SSID argument");
                    }
                }
            }
            // Catch-all for other /strip/ messages for debugging
            else if msg.addr.starts_with("/strip/") { // Note the 'else if'
                    tracing::debug!("Received other Ardour /strip/ feedback: {} {:?}", msg.addr, msg.args);
                }
            // Generic unhandled message log (optional)
            // else {
                    // tracing::debug!("Received unhandled OSC message: {} {:?}", msg.addr, msg.args);
            // }
        }
        osc::Packet::Bundle(bundle) => {
            tracing::debug!("Received OSC bundle from {}:", peer_addr);
            for p_in_bundle in bundle.content {
                Box::pin(handle_osc_packet(p_in_bundle.into(), peer_addr, Arc::clone(&state), Arc::clone(&pending))).await;
            }
        }
    }
} 

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::RawContent;

    fn setup_test_service() -> ArdourService {
        ArdourService::new().expect("Failed to create ArdourService for test")
    }

    #[test]
    fn test_get_server_info() {
        let service = setup_test_service();
        let info = service.get_info();

        assert_eq!(info.server_info.name, "Ardour MCP Server (Rust)");
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION").to_string());
        
        assert_eq!(info.protocol_version, ProtocolVersion::V_2024_11_05);
        assert!(info.capabilities.tools.is_some(), "Tools capability should be Some");
        assert_eq!(info.instructions, Some("This server controls Ardour DAW transport functions.".to_string()));
    }

    #[tokio::test]
    async fn test_transport_play_tool_reports_success_or_osc_error() {
        let service = setup_test_service();
        let result = service.transport_play_tool().await;

        assert!(result.is_ok(), "transport_play_tool call itself failed (should return Ok(CallToolResult) even on OSC error): {:?}", result.err());
        if let Ok(call_result) = result {
            if call_result.is_error == Some(true) {
                tracing::warn!("transport_play_tool reported an OSC error (this is acceptable for the test if Ardour is not perfectly reachable): {:?}", call_result.content);
                assert!(!call_result.content.is_empty(), "Error CallToolResult should have content explaining the error.");
            } else {
                assert!(!call_result.content.is_empty(), "Successful CallToolResult.content was empty");
                let contents = &call_result.content;
                assert_eq!(contents.len(), 1, "Expected one content item for success");
                let content_item = contents.get(0).expect("Failed to get content item for success");
                match &content_item.raw {
                    RawContent::Text(text_val) => {
                        assert_eq!(text_val.text, "Playback started");
                    }
                    other => {
                        panic!("Expected RawContent::Text for success, got {:?}", other);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_set_track_mute_tool_arg_parsing() {
        let service = setup_test_service(); 

        let args_struct = SetTrackMuteArgs { rid: 1, mute_state: true };
        let result = service.set_track_mute_tool(args_struct).await;
        assert!(result.is_ok());
        if let Ok(call_result) = result {
             if call_result.is_error == Some(true) {
                tracing::warn!("set_track_mute_tool reported an OSC error, args might be fine but send failed: {:?}", call_result.content);
            } else {
                assert!(!call_result.content.is_empty(), "Successful mute should have descriptive content.");
            }
        }
    }

     #[tokio::test]
    async fn test_set_transport_speed_tool_arg_validation() {
        let service = setup_test_service();

        let args_out_of_range = SetTransportSpeedArgs { speed: 10.0 }; 
        let result_oor = service.set_transport_speed_tool(args_out_of_range).await;
        assert!(result_oor.is_err(), "Expected an McpError for out-of-range speed, but got Ok");

        let args_in_range = SetTransportSpeedArgs { speed: 1.5 };
        let result_ir = service.set_transport_speed_tool(args_in_range).await;
        assert!(result_ir.is_ok(), "Expected Ok(CallToolResult) for in-range speed, but got McpError");
         if let Ok(call_result) = result_ir {
             if call_result.is_error == Some(true) {
                tracing::warn!("set_transport_speed_tool (in range) reported an OSC error, args were fine but send failed: {:?}", call_result.content);
            } else {
                assert!(!call_result.content.is_empty(), "Successful speed set should have descriptive content.");
            }
        }
    }
} 

