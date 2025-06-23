use crate::tools::AiTool;
use anyhow::anyhow;
use reqwest;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parameters for the DuckDuckGo search tool.
#[derive(Deserialize)]
struct SearchParams {
    /// The search query to send to DuckDuckGo.
    query: String,
}

/// Represents a single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResult {
    title: String,
    link: String,
    snippet: String,
}

/// Tool for searching DuckDuckGo.
pub struct DDGSearchTool;

#[async_trait::async_trait]
impl AiTool for DDGSearchTool {
    fn name(&self) -> &str {
        "ddg_search"
    }

    fn description(&self) -> &str {
        r#"Searches the web using DuckDuckGo.
Important search operators:
cats dogs	results about cats or dogs
"cats and dogs"	exact term (avoid unless necessary)
~"cats and dogs"	semantically similar terms
cats -dogs	reduce results about dogs
cats +dogs	increase results about dogs
cats filetype:pdf	search pdfs about cats (supports doc(x), xls(x), ppt(x), html)
dogs site:example.com	search dogs on example.com
cats -site:example.com	exclude example.com from results
intitle:dogs	title contains "dogs"
inurl:cats	URL contains "cats"

Usage: { \"query\": \"rust async traits\" }"#
    }

    async fn execute(&self, args: &Value) -> anyhow::Result<Value> {
        let params: SearchParams = serde_json::from_value(args.clone())
            .map_err(|_| anyhow!("Missing or invalid 'query' parameter"))?;
        let client = reqwest::Client::new();
        let url = format!("https://duckduckgo.com/html/?q={}", params.query);
        let resp = client
            .get(&url)
            .header("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
            .header("user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:139.0) Gecko/20100101 Firefox/139.0")
            .send()
            .await
            .map_err(|e| anyhow!("Request error: {}", e))?;
        let body = resp
            .text()
            .await
            .map_err(|e| anyhow!("Body error: {}", e))?;
        let document = Html::parse_document(&body);

        let result_selector = Selector::parse(".web-result").unwrap();
        let result_title_selector = Selector::parse(".result__a").unwrap();
        let result_url_selector = Selector::parse(".result__url").unwrap();
        let result_snippet_selector = Selector::parse(".result__snippet").unwrap();

        let results = document
            .select(&result_selector)
            .filter_map(|result| {
                let title = result
                    .select(&result_title_selector)
                    .next()
                    .map(|n| n.text().collect::<Vec<_>>().join(""))
                    .unwrap_or_default();
                let link = result
                    .select(&result_url_selector)
                    .next()
                    .map(|n| n.text().collect::<Vec<_>>().join("").trim().to_string())
                    .unwrap_or_default();
                let snippet = result
                    .select(&result_snippet_selector)
                    .next()
                    .map(|n| n.text().collect::<Vec<_>>().join(""))
                    .unwrap_or_default();

                if !title.is_empty() && !link.is_empty() {
                    Some(SearchResult {
                        title,
                        link,
                        snippet,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        Ok(serde_json::to_value(&results)?)
    }
}
