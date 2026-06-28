//! maia-httpd application.
//!
//! This module contains a top-level structure [`App`] that represents the whole
//! maia-httpd application and a structure [`AppState`] that contains the
//! application state.

use crate::{
    airband::{Airband, AirbandConfig},
    args::Args,
    fpga::{InterruptHandler, IpCore},
    httpd::{self, RecorderFinishWaiter, RecorderState},
    iio::Ad9361,
    spectrometer::{Spectrometer, SpectrometerConfig},
};
use anyhow::Result;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::{
    sync::broadcast,
    time::{Duration, Instant, sleep},
};

/// maia-httpd application.
///
/// This struct represents the maia-sdr application. It owns the different
/// objects of which the application is formed, and runs them concurrently.
#[derive(Debug)]
pub struct App {
    httpd: httpd::Server,
    interrupt_handler: InterruptHandler,
    recorder_finish: RecorderFinishWaiter,
    spectrometer: Spectrometer,
    airband: Option<Airband>,
}

impl App {
    /// Creates a new application.
    #[tracing::instrument(name = "App::new", level = "debug")]
    pub async fn new(args: &Args) -> Result<App> {
        // Initialize and build application state
        let (ip_core, interrupt_handler) = IpCore::take().await?;
        let ip_core = std::sync::Mutex::new(ip_core);
        let ad9361 = tokio::sync::Mutex::new(Ad9361::new().await?);
        let recorder = RecorderState::new(&ad9361, &ip_core).await?;
        // Load the airband config once at startup. It is shared between the
        // running receiver (`Airband`) and the `/api/airband` HTTP handlers,
        // which compare the persisted config against this snapshot to report
        // whether a restart is needed to apply pending changes.
        let airband_config_path = args.airband_config.clone();
        let airband_running = if args.airband {
            Some(AirbandConfig::load(airband_config_path.as_deref()).await?)
        } else {
            None
        };
        let state = AppState(Arc::new(State {
            ad9361,
            ip_core,
            geolocation: std::sync::Mutex::new(None),
            recorder,
            spectrometer_config: Default::default(),
            airband_running,
            airband_config_path,
            // When the airband receiver is enabled it owns the AD9361 front-end
            // (LO/Fs/bandwidth/gain): the channelizer NCO words and decimation
            // are baked for that exact configuration, so the front-end must stay
            // locked. This flag makes the AD9361 HTTP API read-only (see
            // `httpd::ad9361`) so the web UI cannot retune the radio off-band.
            airband_locked: args.airband,
        }));
        // Initialize spectrometer sample rate and mode
        state.spectrometer_config().set_samp_rate_mode(
            state.ad9361().lock().await.get_sampling_frequency().await? as f32,
            state.ip_core().lock().unwrap().spectrometer_mode(),
        );

        // Build application objects

        let (waterfall_sender, _) = broadcast::channel(16);
        let spectrometer = Spectrometer::new(
            state.clone(),
            interrupt_handler.waiter_spectrometer(),
            waterfall_sender.clone(),
        );

        let recorder_finish =
            RecorderFinishWaiter::new(state.clone(), interrupt_handler.waiter_recorder());

        let airband = Airband::new(state.clone(), args).await?;

        let httpd = httpd::Server::new(
            args.listen,
            args.listen_https,
            args.ssl_cert.as_ref(),
            args.ssl_key.as_ref(),
            args.ca_cert.as_ref(),
            state,
            waterfall_sender,
        )
        .await?;

        Ok(App {
            httpd,
            interrupt_handler,
            recorder_finish,
            spectrometer,
            airband,
        })
    }

    /// Runs the application.
    ///
    /// This only returns if one of the objects that form the application fails.
    #[tracing::instrument(name = "App::run", level = "debug", skip_all)]
    pub async fn run(self) -> Result<()> {
        let App {
            httpd,
            interrupt_handler,
            recorder_finish,
            spectrometer,
            airband,
        } = self;
        // The airband receiver is optional and must never bring down the rest of
        // maia-httpd. When disabled this future pends forever; when enabled it
        // supervises the receiver in-process, restarting it with capped
        // exponential backoff if it fails (e.g. a detected DMA stall) so the
        // data path self-heals without killing the web UI / spectrometer.
        let airband = async move {
            if let Some(a) = airband {
                let mut backoff = Duration::from_secs(1);
                loop {
                    let started = Instant::now();
                    match a.run().await {
                        Ok(()) => tracing::warn!("airband receiver exited cleanly; restarting"),
                        Err(e) => tracing::error!("airband receiver failed: {e:#}; restarting"),
                    }
                    // A run that stayed up a while was healthy: reset the backoff
                    // so a later transient failure recovers quickly.
                    if started.elapsed() >= Duration::from_secs(60) {
                        backoff = Duration::from_secs(1);
                    }
                    tracing::info!("airband receiver restarting in {backoff:?}");
                    sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
            std::future::pending::<Result<()>>().await
        };
        tokio::select! {
            ret = httpd.run() => ret,
            ret = interrupt_handler.run() => ret,
            ret = recorder_finish.run() => ret,
            ret = spectrometer.run() => ret,
            ret = airband => ret,
        }
    }
}

/// Application state.
///
/// This struct contains the application state that needs to be shared between
/// different modules, such as different Axum handlers in the HTTP server. The
/// struct behaves as an `Arc<...>`. It is cheaply clonable and clones represent
/// a reference to a shared object.
#[derive(Debug, Clone)]
pub struct AppState(Arc<State>);

#[derive(Debug)]
struct State {
    ad9361: tokio::sync::Mutex<Ad9361>,
    ip_core: Mutex<IpCore>,
    geolocation: Mutex<Option<maia_json::Geolocation>>,
    recorder: RecorderState,
    spectrometer_config: SpectrometerConfig,
    airband_locked: bool,
    airband_running: Option<AirbandConfig>,
    airband_config_path: Option<PathBuf>,
}

impl AppState {
    /// Gives access to the [`Ad9361`] object of the application.
    pub fn ad9361(&self) -> &tokio::sync::Mutex<Ad9361> {
        &self.0.ad9361
    }

    /// Gives access to the [`IpCore`] object of the application.
    pub fn ip_core(&self) -> &Mutex<IpCore> {
        &self.0.ip_core
    }

    /// Gives access to the current geolocation of the device.
    ///
    /// The geolocation is `None` if it has never been set or if it has been
    /// cleared, or a valid [`Geolocation`](maia_json::Geolocation) otherwise.
    pub fn geolocation(&self) -> &Mutex<Option<maia_json::Geolocation>> {
        &self.0.geolocation
    }

    /// Gives access to the [`RecorderState`] object of the application.
    pub fn recorder(&self) -> &RecorderState {
        &self.0.recorder
    }

    /// Gives access to the [`SpectrometerConfig`] object of the application.
    pub fn spectrometer_config(&self) -> &SpectrometerConfig {
        &self.0.spectrometer_config
    }

    /// Returns the AD9361 sampling frequency.
    pub async fn ad9361_samp_rate(&self) -> Result<f64> {
        Ok(self.ad9361().lock().await.get_sampling_frequency().await? as f64)
    }

    /// Returns whether the AD9361 front-end is locked by the airband receiver.
    ///
    /// When this is `true` the airband multichannel receiver owns the AD9361
    /// configuration and the AD9361 HTTP API (`/api/ad9361`) is read-only, so
    /// that the web UI cannot retune the front-end away from the airband band.
    pub fn airband_locked(&self) -> bool {
        self.0.airband_locked
    }

    /// Returns the airband config loaded at startup (the running plan).
    ///
    /// This is `None` when the airband receiver is disabled. The `/api/airband`
    /// handlers compare the persisted config against this snapshot to report
    /// whether a restart is needed to apply changes.
    pub fn airband_running(&self) -> Option<&AirbandConfig> {
        self.0.airband_running.as_ref()
    }

    /// Returns the path to the airband JSON config file, if configured.
    ///
    /// This is where `/api/airband` PATCH requests persist the channel plan and
    /// front-end settings.
    pub fn airband_config_path(&self) -> Option<&Path> {
        self.0.airband_config_path.as_deref()
    }
}
