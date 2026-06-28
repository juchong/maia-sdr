//! `/api/health` handler.
//!
//! A lightweight internal health view of the airband data path, derived purely
//! from the FPGA IP core. The host-side reader already answers the operator's
//! three questions (Pluto reachable / stream up / data flowing) on its own, so
//! this endpoint is optional enrichment: it exposes deeper *internal* Pluto
//! state (live DMA progress and the overflow flag) that the reader can scrape
//! and republish if desired.
//!
//!   * `airband_enabled` — the airband receiver owns the front-end.
//!   * `dma_address` / `dma_advancing` — the FPGA write pointer and whether it
//!     moved across a short sample window (the data path is producing samples).
//!   * `overflow` — the FPGA overflow flag (host reader not keeping up).

use crate::app::AppState;
use axum::{Json, extract::State};
use serde_json::{Value, json};
use tokio::time::{Duration, sleep};

pub async fn get_health(State(state): State<AppState>) -> Json<Value> {
    let (addr1, overflow) = {
        let core = state.ip_core().lock().unwrap();
        (core.airband_next_address(), core.airband_overflow())
    };
    // Sample the write pointer twice to tell "advancing" from "stalled".
    sleep(Duration::from_millis(50)).await;
    let addr2 = state.ip_core().lock().unwrap().airband_next_address();

    Json(json!({
        "airband_enabled": state.airband_locked(),
        "dma_address": addr2,
        "dma_advancing": addr2 != addr1,
        "overflow": overflow,
    }))
}
