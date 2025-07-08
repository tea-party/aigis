use anyhow::Result;

mod cli;
use cli::run_cli;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    run_cli().await
}
