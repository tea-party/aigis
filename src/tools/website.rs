use anyhow::{Error, anyhow};
use serde_json::Value;
use tracing::debug;

use crate::tools::AiTool;

/// Example tool that performs basic math operations.
pub struct WebsiteTool;

#[async_trait::async_trait]
impl AiTool for WebsiteTool {
    fn name(&self) -> &str {
        "website"
    }

    fn description(&self) -> &str {
        r#"Fetches a website.
Parameters:
- `website`: The URL of the website to fetch.
- `render`: Which format to render the content in. Options are "html" or "md" (default is "md").
Example usage: { "website": "https://example.com", "render": "md"}
"#
    }

    async fn execute(&self, args: &Value) -> Result<Value, Error> {
        let client = reqwest::Client::new();
        let website = args
            .get("website")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'website' parameter"))?;
        let render = args.get("render").and_then(|v| v.as_str()).unwrap_or("md");

        let resp = client
            .get(website)
            .send()
            .await
            .map_err(|e| anyhow!("Request error: {}", e))?;

        debug!("Response status: {}", resp.status());

        let body = resp
            .text()
            .await
            .map_err(|e| anyhow!("Body error: {}", e))?;

        debug!("Response body length: {}", body.len());

        if render == "html" {
            Ok(serde_json::json!({ "content": body }))
        } else if render == "md" {
            let markdown = html2md::rewrite_html(&body, false);
            debug!("Converted HTML to Markdown, length: {}", markdown.len());
            Ok(serde_json::json!({ "content": markdown }))
        } else {
            Err(anyhow!(
                "Invalid 'render' parameter, must be 'html' or 'md'"
            ))
        }
    }
}
