//! `/api/airband` handlers.
//!
//! These handlers expose the airband multichannel receiver channel plan and
//! front-end configuration to the web config UI (`/airband.html`).
//!
//!   * `GET /api/airband` returns the persisted (pending) configuration plus
//!     capability fields and a `needs_restart` flag.
//!   * `PATCH /api/airband` validates and persists changes to the config file.
//!
//! Changes do not affect the running receiver until maia-httpd is restarted
//! (`POST /api/system/restart`), since the channelizer NCO words are programmed
//! once at startup from this config.

use super::json_error::JsonError;
use crate::{
    airband::{AirbandConfig, N_CHANNELS},
    app::AppState,
};
use anyhow::{Result, anyhow};
use axum::{Json, extract::State};
use maia_json::{Airband, AirbandAgcMode, AirbandChannel, PatchAirband};

fn agc_to_mode(agc: &Option<String>) -> AirbandAgcMode {
    match agc.as_deref() {
        Some("slow_attack") => AirbandAgcMode::SlowAttack,
        Some("fast_attack") => AirbandAgcMode::FastAttack,
        Some("hybrid") => AirbandAgcMode::Hybrid,
        _ => AirbandAgcMode::Manual,
    }
}

fn mode_to_agc(mode: AirbandAgcMode) -> String {
    match mode {
        AirbandAgcMode::Manual => "manual",
        AirbandAgcMode::SlowAttack => "slow_attack",
        AirbandAgcMode::FastAttack => "fast_attack",
        AirbandAgcMode::Hybrid => "hybrid",
    }
    .to_string()
}

fn config_to_json(cfg: &AirbandConfig, enabled: bool, needs_restart: bool) -> Airband {
    let labels = cfg.channel_labels.clone().unwrap_or_default();
    let channels = cfg
        .channels_hz
        .iter()
        .enumerate()
        .map(|(i, &freq_hz)| AirbandChannel {
            freq_hz,
            label: labels.get(i).cloned().filter(|s| !s.is_empty()),
        })
        .collect();
    Airband {
        enabled,
        center_hz: cfg.center_hz,
        samp_rate: cfg.samp_rate,
        rf_bandwidth: cfg.rf_bandwidth,
        gain_db: cfg.gain_db,
        agc: agc_to_mode(&cfg.agc),
        channels,
        poll_ms: cfg.poll_ms,
        max_channels: N_CHANNELS as u32,
        samp_rate_locked: true,
        needs_restart,
    }
}

/// Returns the pending configuration: the on-disk config file if present,
/// otherwise the running config, otherwise the built-in default.
async fn pending_config(state: &AppState) -> Result<AirbandConfig> {
    if let Some(path) = state.airband_config_path() {
        match tokio::fs::read_to_string(path).await {
            Ok(s) => {
                return serde_json::from_str(&s)
                    .map_err(|e| anyhow!("failed to parse airband config {path:?}: {e}"));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(anyhow!("failed to read airband config {path:?}: {e}")),
        }
    }
    Ok(state.airband_running().cloned().unwrap_or_default())
}

fn needs_restart(state: &AppState, pending: &AirbandConfig) -> bool {
    match state.airband_running() {
        Some(running) => pending != running,
        // Receiver disabled: nothing is running to differ from.
        None => false,
    }
}

pub async fn get_airband(State(state): State<AppState>) -> Result<Json<Airband>, JsonError> {
    let pending = pending_config(&state)
        .await
        .map_err(JsonError::server_error)?;
    let needs_restart = needs_restart(&state, &pending);
    Ok(Json(config_to_json(
        &pending,
        state.airband_locked(),
        needs_restart,
    )))
}

fn apply_and_validate(cfg: &mut AirbandConfig, patch: &PatchAirband) -> Result<()> {
    if let Some(center_hz) = patch.center_hz {
        cfg.center_hz = center_hz;
    }
    if let Some(rf_bandwidth) = patch.rf_bandwidth {
        cfg.rf_bandwidth = Some(rf_bandwidth);
    }
    if let Some(gain_db) = patch.gain_db {
        anyhow::ensure!(
            (0.0..=77.0).contains(&gain_db),
            "gain_db {gain_db} is out of range (0..=77 dB)"
        );
        cfg.gain_db = gain_db;
    }
    if let Some(agc) = patch.agc {
        cfg.agc = Some(mode_to_agc(agc));
    }
    if let Some(poll_ms) = patch.poll_ms {
        anyhow::ensure!(poll_ms >= 1, "poll_ms must be at least 1 ms");
        cfg.poll_ms = poll_ms;
    }
    if let Some(channels) = &patch.channels {
        anyhow::ensure!(
            channels.len() <= N_CHANNELS,
            "too many channels: {} (max {})",
            channels.len(),
            N_CHANNELS
        );
        // Channels must fall inside the capture window [center - Fs/2, center + Fs/2).
        let half = f64::from(cfg.samp_rate) / 2.0;
        let center = cfg.center_hz as f64;
        let mut freqs = Vec::with_capacity(channels.len());
        let mut labels = Vec::with_capacity(channels.len());
        let mut any_label = false;
        for ch in channels {
            let offset = ch.freq_hz - center;
            anyhow::ensure!(
                (-half..half).contains(&offset),
                "channel {:.4} MHz is outside the capture window (center {:.4} MHz, +/-{:.3} MHz)",
                ch.freq_hz / 1e6,
                center / 1e6,
                half / 1e6
            );
            freqs.push(ch.freq_hz);
            let label = ch.label.clone().unwrap_or_default();
            if !label.is_empty() {
                any_label = true;
            }
            labels.push(label);
        }
        cfg.channels_hz = freqs;
        cfg.channel_labels = if any_label { Some(labels) } else { None };
    }
    Ok(())
}

pub async fn patch_airband(
    State(state): State<AppState>,
    Json(patch): Json<PatchAirband>,
) -> Result<Json<Airband>, JsonError> {
    let mut cfg = pending_config(&state)
        .await
        .map_err(JsonError::server_error)?;
    apply_and_validate(&mut cfg, &patch).map_err(JsonError::client_error_alert)?;

    let path = state
        .airband_config_path()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| JsonError::client_error_alert(anyhow!("no airband config path configured")))?;
    let json = serde_json::to_string_pretty(&cfg).map_err(JsonError::server_error)?;
    tokio::fs::write(&path, json)
        .await
        .map_err(|e| JsonError::server_error(anyhow!("failed to write {path:?}: {e}")))?;

    let needs_restart = needs_restart(&state, &cfg);
    Ok(Json(config_to_json(
        &cfg,
        state.airband_locked(),
        needs_restart,
    )))
}
