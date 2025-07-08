use anyhow::{Error, anyhow};
use regex::Regex;
use serde_json::Value;
use tracing::info;

pub mod calc;
pub mod search;
pub mod website;

#[async_trait::async_trait]
pub trait AiTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn execute(&self, args: &Value) -> anyhow::Result<Value>;
}

/// Represents a parsed tool call from an LLM response.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool_type: String,
    pub tool_name: String,
    pub tool_args: serde_json::Value,
}

/// Parses all tool calls from a response string using the new special format.
/// For each function call, expects this format:
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
/// ```json
/// {...}
/// ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
pub fn parse_tool_calls(response: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let re = Regex::new(
        r"<\u{FF5C}tool▁call▁begin\u{FF5C}>(?P<type>\w+)<\u{FF5C}tool▁sep\u{FF5C}>(?P<name>\w+)\s*```json\s*(?P<args>\{.*?\})\s*```<\u{FF5C}tool▁call▁end\u{FF5C}>"
).unwrap();

    for cap in re.captures_iter(response) {
        let tool_type = cap
            .name("type")
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "function".to_string());
        let tool_name = cap
            .name("name")
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        let args_str = cap.name("args").map(|m| m.as_str()).unwrap_or("{}");
        if let Ok(tool_args) = serde_json::from_str(args_str) {
            calls.push(ToolCall {
                tool_type: tool_type.clone(),
                tool_name,
                tool_args,
            });
        }
    }
    calls
}

/// Executes a list of tool calls using the provided tools.
/// Returns a Vec of (tool_name, result or error string).
pub async fn execute_tool_calls(
    tool_calls: &[ToolCall],
    tools: &[Box<dyn AiTool>],
) -> Vec<(String, Result<Value, String>)> {
    let mut results = Vec::new();
    info!("Executing {} tool calls", tool_calls.len());
    for call in tool_calls {
        if let Some(tool) = tools.iter().find(|t| t.name() == call.tool_name) {
            info!("Executing tool: {}", call.tool_name);
            match tool.execute(&call.tool_args).await {
                Ok(res) => results.push((call.tool_name.clone(), Ok(res))),
                Err(e) => results.push((call.tool_name.clone(), Err(format!("Error: {}", e)))),
            }
        } else {
            results.push((call.tool_name.clone(), Err("Tool not found".to_string())));
        }
    }
    results
}
