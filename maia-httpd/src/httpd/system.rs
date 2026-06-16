//! `/api/system` handlers.
//!
//! Currently only `POST /api/system/restart`, which restarts the maia-httpd
//! service so that a freshly saved airband config is applied (the channelizer
//! NCO words are programmed once at startup from the config file).

use super::json_error::JsonError;
use axum::Json;
use serde_json::{Value, json};

/// Init script that supervises maia-httpd on the PlutoSDR rootfs.
const INIT_SCRIPT: &str = "/etc/init.d/S60maia-httpd";

/// Restarts the maia-httpd service after a short delay.
///
/// The restart is detached into a child shell with a one second delay so this
/// HTTP response can be flushed before the current process is replaced. The
/// init script kills maia-httpd by name (not by process group), so the detached
/// shell survives the stop and brings the service back up with the new config.
/// If the init script is unavailable, the client should fall back to a manual
/// reboot (e.g. `dfu-util -a firmware.dfu -e`).
pub async fn post_restart() -> Result<Json<Value>, JsonError> {
    std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("sleep 1; {INIT_SCRIPT} restart"))
        .spawn()
        .map_err(JsonError::server_error)?;
    Ok(Json(json!({ "restarting": true })))
}
