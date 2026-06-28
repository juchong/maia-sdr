//! Airband multichannel receiver control and framed-audio streaming.
//!
//! This module drives the airband multichannel receiver that is spliced into
//! the Maia SDR FPGA (`maia_hdl` `ReceiverTop` + cyclic `DmaStreamWrite`). It
//!
//!   1. configures the AD9361 front-end (sample rate, RX LO, bandwidth, gain),
//!   2. programs the per-channel NCO tuning words from a channel plan,
//!   3. enables the receiver and starts the cyclic audio DMA, and
//!   4. continuously drains the framed-audio DDR ring and streams the raw
//!      64-bit records to TCP clients (e.g. a Raspberry Pi feeder).
//!
//! The DDR ring is exposed by the `maia-sdr.ko` `rxbuffer` device
//! (`/dev/maia-sdr-airband`); see [`crate::rxbuffer`]. Each record is a
//! little-endian 64-bit word `{ seq[63:40] | chan[39:32] | sample[31:0] }`
//! (see `maia_hdl` `AudioFramer`). The stream is left as raw records so that
//! the demux, per-channel sequence/gap handling and audio scaling all live in
//! the host reader.

use crate::{app::AppState, args::Args};
use anyhow::{Context, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::Path};
use tokio::{
    io::AsyncWriteExt,
    net::{TcpListener, TcpStream},
    sync::broadcast,
    time::{Duration, Instant, sleep},
};

/// Size in bytes of one framed audio record (`maia_hdl` `AudioFramer`).
pub const FRAME_BYTES: usize = 8;

/// The FPGA writes the audio ring continuously (silence still produces samples),
/// so a write pointer that stops advancing for this long means the DMA/FPGA has
/// stalled. The reader fails loudly so the supervisor restarts the data path
/// instead of streaming a frozen buffer forever.
const DMA_STALL_TIMEOUT: Duration = Duration::from_secs(5);

/// Sustained FPGA overflow (the host reader cannot keep up draining the ring)
/// for this long is escalated from a one-shot warning to a hard failure.
const OVERFLOW_ESCALATE: Duration = Duration::from_secs(15);

/// Number of channels instantiated in the FPGA receiver (`maia_hdl`
/// `_AIRBAND_N_CHANNELS`).
pub const N_CHANNELS: usize = 18;

/// NCO tuning-word width in the FPGA receiver (`maia_hdl` `_AIRBAND_NCO_WIDTH`).
const NCO_WIDTH: u32 = 24;

/// Broadcast channel depth (chunks). Slow clients that fall behind by more than
/// this are lagged (and will observe a per-channel sequence jump).
const BROADCAST_DEPTH: usize = 256;

/// Airband receiver configuration.
///
/// Deserialized from the optional JSON config file; falls back to
/// [`AirbandConfig::default`] when absent. `samp_rate` MUST match the sample
/// rate assumed when computing the channelizer (audio rate =
/// `samp_rate / lane_decim / audio_decim` = `samp_rate / 160 / 5` = 20000 sps at
/// the 16 MHz build).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct AirbandConfig {
    /// AD9361 RX LO center frequency (Hz).
    pub center_hz: u64,
    /// AD9361 sample rate (Hz).
    pub samp_rate: u32,
    /// AD9361 RX RF bandwidth (Hz). Defaults to `samp_rate` when `None`.
    pub rf_bandwidth: Option<u32>,
    /// Manual RX gain (dB); used when `agc` is `None` or `"manual"`.
    pub gain_db: f64,
    /// AGC mode: `"manual"`, `"slow_attack"`, `"fast_attack"`, `"hybrid"`.
    pub agc: Option<String>,
    /// Absolute channel center frequencies (Hz). At most [`N_CHANNELS`] are used.
    pub channels_hz: Vec<f64>,
    /// Optional per-channel labels, parallel to `channels_hz`.
    ///
    /// Purely cosmetic (used by the web config UI); ignored by the receiver.
    /// Absent in older config files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_labels: Option<Vec<String>>,
    /// Ring poll interval in milliseconds.
    pub poll_ms: u64,
}

impl Default for AirbandConfig {
    fn default() -> AirbandConfig {
        // The operational channel plan + gain live in an external config file
        // (firmware/airband.json on the Pluto's SD card, loaded via
        // `--airband-config /mnt/sdcard/airband.json`). This built-in default is
        // a deliberately MINIMAL FALLBACK, used only when that file is absent
        // (e.g. the SD card is missing, unformatted, or unmounted): a single
        // channel -- 118.050 MHz S50 AWOS, an always-on carrier -- at 0 dB gain.
        AirbandConfig {
            // samp_rate MUST match the rate the channelizer was built for (16 MHz
            // -> 20000 sps audio). The fallback LO (123.438 MHz) is chosen to keep
            // the 118.050 indicator comfortably inside +/- Fs/2; the OPERATIONAL
            // capture (SD airband.json) re-centers to 126.4 MHz to admit 133.65 MHz
            // (see hdl/capture_window.py).
            center_hz: 123_438_000,
            samp_rate: 16_000_000,
            rf_bandwidth: None,
            // 0 dB is intentional for the fallback. The receiver is
            // INTERNAL-noise-limited; at 0 dB a weak airband carrier sits AT the
            // ADC quantization floor (a controlled sweep on the continuous
            // 118.050 AWOS carrier measured audio SNR ~1 dB at 0 dB, rising to
            // ~10-12 dB by 6-12 dB), so this fallback is near-silent BY DESIGN:
            // hearing only a faint AWOS (and just one channel) is the obvious cue
            // that the SD channel plan did not load. The real operating gain
            // (~12 dB with an external LNA; toward the 48 dB clipping knee on a
            // bare front end) is set in the SD config -- see firmware/airband.json
            // and SPUR-INVESTIGATION.md.
            gain_db: 0.0,
            agc: Some("manual".to_string()),
            channels_hz: vec![118_050_000.0],
            channel_labels: Some(vec!["S50 AWOS".to_string()]),
            poll_ms: 20,
        }
    }
}

impl AirbandConfig {
    /// Loads the configuration from `path` (JSON), or returns the default
    /// configuration if `path` is `None` or the file does not exist.
    pub async fn load(path: Option<&Path>) -> Result<AirbandConfig> {
        let Some(path) = path else {
            return Ok(AirbandConfig::default());
        };
        match tokio::fs::read_to_string(path).await {
            Ok(s) => {
                let cfg: AirbandConfig = serde_json::from_str(&s)
                    .with_context(|| format!("failed to parse airband config {path:?}"))?;
                Ok(cfg)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!("airband config {path:?} not found; using defaults");
                Ok(AirbandConfig::default())
            }
            Err(e) => Err(e).with_context(|| format!("failed to read airband config {path:?}")),
        }
    }
}

/// Airband receiver task.
#[derive(Debug)]
pub struct Airband {
    state: AppState,
    config: AirbandConfig,
    listen: SocketAddr,
}

impl Airband {
    /// Builds the airband task from the CLI arguments, or returns `None` if the
    /// airband receiver is not enabled (it is opt-in via `--airband`).
    pub async fn new(state: AppState, args: &Args) -> Result<Option<Airband>> {
        if !args.airband {
            tracing::info!("airband receiver disabled (pass --airband to enable)");
            return Ok(None);
        }
        // Reuse the config already loaded into the application state (so the
        // running plan and the HTTP `/api/airband` view share one source of
        // truth); fall back to a fresh load if it is somehow absent.
        let config = match state.airband_running() {
            Some(config) => config.clone(),
            None => AirbandConfig::load(args.airband_config.as_deref()).await?,
        };
        Ok(Some(Airband {
            state,
            config,
            listen: args.airband_listen,
        }))
    }

    /// Configures the front-end and DSP, then streams framed audio forever.
    ///
    /// Only returns on error (e.g. failure to open the DMA buffer, bind the TCP
    /// listener, or a detected DMA stall / sustained overflow). Takes `&self` so
    /// [`crate::app::App::run`] can re-invoke it to restart the data path after a
    /// failure (each call re-configures the front-end and re-binds the socket).
    pub async fn run(&self) -> Result<()> {
        self.configure().await.context("airband configuration failed")?;

        let buffer = crate::rxbuffer::RxBuffer::new("maia-sdr-airband")
            .await
            .context("failed to open maia-sdr-airband DMA buffer")?;

        let (tx, _rx) = broadcast::channel::<Bytes>(BROADCAST_DEPTH);

        let listener = TcpListener::bind(self.listen)
            .await
            .with_context(|| format!("failed to bind airband TCP {}", self.listen))?;
        tracing::info!(
            "airband framed-audio stream listening on tcp://{} ({} channels, audio {:.1} sps)",
            self.listen,
            N_CHANNELS,
            self.config.samp_rate as f64 / 160.0 / 5.0,
        );

        let accept_tx = tx.clone();
        let server = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((sock, peer)) => {
                        tracing::info!("airband client connected: {peer}");
                        let rx = accept_tx.subscribe();
                        tokio::spawn(client_task(sock, peer, rx));
                    }
                    Err(e) => tracing::warn!("airband accept error: {e}"),
                }
            }
        });

        let res = self.reader_loop(&buffer, &tx).await;
        server.abort();
        res
    }

    /// Configures the AD9361 front-end, programs the NCOs, enables the receiver
    /// and starts the cyclic audio DMA.
    async fn configure(&self) -> Result<()> {
        let fs = self.config.samp_rate as f64;
        {
            let ad9361 = self.state.ad9361().lock().await;
            ad9361.set_sampling_frequency(self.config.samp_rate).await?;
            ad9361
                .set_rx_rf_bandwidth(self.config.rf_bandwidth.unwrap_or(self.config.samp_rate))
                .await?;
            ad9361.set_rx_lo_frequency(self.config.center_hz).await?;
            match self.config.agc.as_deref() {
                Some("manual") | None => {
                    ad9361
                        .set_rx_gain_mode(maia_json::Ad9361GainMode::Manual.into())
                        .await?;
                    ad9361.set_rx_gain(self.config.gain_db).await?;
                }
                Some(mode) => {
                    let mode: crate::iio::Ad9361GainMode = mode
                        .parse()
                        .map_err(|()| anyhow::anyhow!("invalid AGC mode {mode:?}"))?;
                    ad9361.set_rx_gain_mode(mode.into()).await?;
                }
            }
        }
        tracing::info!(
            "airband front-end: LO {:.3} MHz, Fs {:.3} Msps, {} channels",
            self.config.center_hz as f64 / 1e6,
            fs / 1e6,
            self.config.channels_hz.len().min(N_CHANNELS),
        );

        let scale = (1u64 << NCO_WIDTH) as f64;
        let mask = (1u32 << NCO_WIDTH) - 1;
        let core = self.state.ip_core().lock().unwrap();
        core.airband_set_enable(false);
        for (i, &f) in self.config.channels_hz.iter().take(N_CHANNELS).enumerate() {
            let offset = (f - self.config.center_hz as f64) / fs; // cycles/sample
            if !(-0.5..0.5).contains(&offset) {
                anyhow::bail!(
                    "channel {f:.3} Hz is outside the +/- Fs/2 capture window \
                     (LO {} Hz, Fs {} Hz)",
                    self.config.center_hz,
                    self.config.samp_rate
                );
            }
            let word = ((offset * scale).round() as i64 as u32) & mask;
            core.airband_set_frequency(i as u8, word);
        }
        core.airband_set_enable(true);
        core.airband_dma_start();
        Ok(())
    }

    /// Drains the framed-audio ring and broadcasts whole, fully-written buffers
    /// to the connected TCP clients.
    async fn reader_loop(
        &self,
        buffer: &crate::rxbuffer::RxBuffer,
        tx: &broadcast::Sender<Bytes>,
    ) -> Result<()> {
        let buf_sz = buffer.buffer_size();
        let num = buffer.num_buffers();
        assert!(
            buf_sz % FRAME_BYTES == 0,
            "airband buffer size {buf_sz} is not a multiple of the {FRAME_BYTES}-byte record"
        );
        let interval = Duration::from_millis(self.config.poll_ms.max(1));

        // Start reading at the buffer the FPGA is currently writing.
        let write_buffer = |core: &crate::fpga::IpCore| (core.airband_next_address() / buf_sz) % num;
        let mut read_buf = write_buffer(&self.state.ip_core().lock().unwrap());
        let mut last_overflow = false;
        let mut overflow_since: Option<Instant> = None;
        let mut prev_write_buf = read_buf;
        let mut last_write_change = Instant::now();

        loop {
            sleep(interval).await;

            let (write_buf, overflow) = {
                let core = self.state.ip_core().lock().unwrap();
                (write_buffer(&core), core.airband_overflow())
            };

            // Overflow: warn once on the edge, escalate to a hard failure if it
            // persists (the reader is not keeping up -> restart the data path).
            if overflow && !last_overflow {
                tracing::warn!("airband FPGA overflow: audio samples were dropped");
            }
            last_overflow = overflow;
            if overflow {
                let since = *overflow_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= OVERFLOW_ESCALATE {
                    anyhow::bail!(
                        "airband FPGA overflow sustained for {OVERFLOW_ESCALATE:?}; \
                         host reader is not draining the ring"
                    );
                }
            } else {
                overflow_since = None;
            }

            // DMA-stall watchdog: a write pointer that has not advanced for
            // DMA_STALL_TIMEOUT means the FPGA/DMA stopped producing samples.
            if write_buf != prev_write_buf {
                prev_write_buf = write_buf;
                last_write_change = Instant::now();
            } else if last_write_change.elapsed() >= DMA_STALL_TIMEOUT {
                anyhow::bail!(
                    "airband DMA stall: FPGA write pointer stuck at buffer {write_buf} \
                     for {DMA_STALL_TIMEOUT:?}"
                );
            }

            // Consume whole buffers that are at least two buffers behind the one
            // being written. Keeping a >= 2-buffer gap guarantees the buffer has
            // fully landed in DDR (the DMA issues address bursts up to two ahead
            // of the committed write data), so we never read in-flight bytes.
            while (write_buf + num - read_buf) % num >= 2 {
                buffer
                    .cache_invalidate(read_buf)
                    .with_context(|| format!("airband cache invalidate buffer {read_buf}"))?;
                let chunk = Bytes::copy_from_slice(buffer.buffer_as_slice(read_buf));
                // Ignore the error when there are no subscribers.
                let _ = tx.send(chunk);
                read_buf = (read_buf + 1) % num;
            }
        }
    }
}

/// Forwards broadcast framed-audio chunks to one connected TCP client.
async fn client_task(mut sock: TcpStream, peer: SocketAddr, mut rx: broadcast::Receiver<Bytes>) {
    let _ = sock.set_nodelay(true);
    loop {
        match rx.recv().await {
            Ok(chunk) => {
                if let Err(e) = sock.write_all(&chunk).await {
                    tracing::info!("airband client {peer} disconnected: {e}");
                    return;
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("airband client {peer} lagged, dropped {n} chunks");
            }
            Err(broadcast::error::RecvError::Closed) => return,
        }
    }
}
