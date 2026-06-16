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
use serde::Deserialize;
use std::{net::SocketAddr, path::Path};
use tokio::{
    io::AsyncWriteExt,
    net::{TcpListener, TcpStream},
    sync::broadcast,
    time::{Duration, sleep},
};

/// Size in bytes of one framed audio record (`maia_hdl` `AudioFramer`).
pub const FRAME_BYTES: usize = 8;

/// Number of channels instantiated in the FPGA receiver (`maia_hdl`
/// `_AIRBAND_N_CHANNELS`).
pub const N_CHANNELS: usize = 21;

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
/// `samp_rate / lane_decim / audio_decim` = `samp_rate / 128 / 7`).
#[derive(Debug, Clone, Deserialize)]
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
    /// Ring poll interval in milliseconds.
    pub poll_ms: u64,
}

impl Default for AirbandConfig {
    fn default() -> AirbandConfig {
        AirbandConfig {
            // Capture window from hdl/capture_window.py: center 123.438 MHz,
            // Fs ~= 14 MHz comfortably covers the channel list below (the
            // 133.65 MHz "nice to have" is outside this window and omitted).
            center_hz: 123_438_000,
            samp_rate: 14_000_000,
            rf_bandwidth: None,
            // Airband voice is weak and intermittent. The AD9361 AGC modes
            // (slow/fast/hybrid) settle to ~55 dB on the wideband power and
            // starve weak narrowband channels (measured: ch0 peak ~5x lower
            // than fixed max gain). Default to fixed manual gain near max; the
            // ADC does not overload at this site (peak sums scale linearly with
            // gain). Lower `gain_db` if a strong local signal causes audible
            // distortion across channels.
            gain_db: 71.0,
            agc: Some("manual".to_string()),
            channels_hz: vec![
                118_050_000.0,
                119_200_000.0,
                119_900_000.0,
                120_100_000.0,
                120_400_000.0,
                120_950_000.0,
                121_500_000.0,
                121_600_000.0,
                121_700_000.0,
                122_275_000.0,
                122_950_000.0,
                122_975_000.0,
                123_900_000.0,
                124_700_000.0,
                125_600_000.0,
                125_900_000.0,
                126_250_000.0,
                126_500_000.0,
                126_875_000.0,
                127_100_000.0,
                128_500_000.0,
            ],
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
        let config = AirbandConfig::load(args.airband_config.as_deref()).await?;
        Ok(Some(Airband {
            state,
            config,
            listen: args.airband_listen,
        }))
    }

    /// Configures the front-end and DSP, then streams framed audio forever.
    ///
    /// Only returns on error (e.g. failure to open the DMA buffer or bind the
    /// TCP listener), so that [`crate::app::App::run`] can treat it as fatal.
    pub async fn run(self) -> Result<()> {
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
            self.config.samp_rate as f64 / 128.0 / 7.0,
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

        loop {
            sleep(interval).await;

            let write_buf = {
                let core = self.state.ip_core().lock().unwrap();
                let of = core.airband_overflow();
                if of && !last_overflow {
                    tracing::warn!("airband FPGA overflow: audio samples were dropped");
                }
                last_overflow = of;
                write_buffer(&core)
            };

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
