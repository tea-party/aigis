#![allow(unused)]

pub mod cursor;
pub mod embed;
pub mod ingestors;
pub mod kv;
pub mod llm;
pub mod tools;
pub mod vdb;

use anyhow::Result;
use atrium_api::types::string::Did;
use bsky_sdk::BskyAgent;
use metrics;
use metrics_exporter_prometheus::PrometheusBuilder;
use once_cell::sync::Lazy;
use tracing::{error, info};

/// Initialize tracing subscriber for logging.
pub fn setup_tracing() {
    tracing_subscriber::fmt::init();
}

static POSTS_INGESTED: Lazy<metrics::Counter> =
    Lazy::new(|| metrics::counter!("posts_ingested_total"));
static INGEST_ERRORS: Lazy<metrics::Counter> =
    Lazy::new(|| metrics::counter!("ingest_errors_total"));
static INGEST_LATENCY: Lazy<metrics::Histogram> =
    Lazy::new(|| metrics::histogram!("ingest_latency_seconds"));

/// Initialize Prometheus metrics exporter.
pub fn setup_metrics() {
    if let Err(e) = PrometheusBuilder::new().install() {
        error!(
            "Failed to install, program will run without Prometheus exporter: {}",
            e
        );
    }
}

/// Set up a Bluesky session and return the agent and DID.
pub async fn setup_bsky_sess() -> Result<(BskyAgent, Did)> {
    let span = tracing::info_span!("setup_bsky_sess");
    let _enter = span.enter();

    let agent = BskyAgent::builder().build().await?;
    let res = agent
        .login(std::env::var("ATP_USER")?, std::env::var("ATP_PASSWORD")?)
        .await?;

    info!("logged in as {}", res.handle.to_string());

    Ok((agent, res.did.to_owned()))
}
