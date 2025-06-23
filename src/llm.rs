use std::pin::Pin;

use crate::tools::AiTool;
use anyhow::{Error, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use genai::chat::{ChatMessage, ChatRequest};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ModelIden, ServiceTarget, adapter::AdapterKind};

#[async_trait]
pub trait AiService {
    async fn generate_response(
        &self,
        messages: &Vec<ChatMessage>,
        searched_messages: Option<&Vec<ChatMessage>>,
    ) -> Result<String, Error>;
    async fn generate_response_stream<'a>(
        &'a self,
        messages: &'a Vec<ChatMessage>,
        searched_messages: Option<&'a Vec<ChatMessage>>,
    ) -> Result<
        Pin<
            Box<
                dyn futures_core::Stream<Item = Result<genai::chat::ChatStreamEvent, anyhow::Error>>
                    + Send
                    + 'a,
            >,
        >,
        anyhow::Error,
    >;
}

pub struct LLMService {
    client: Client,
    system_prompt: Option<String>,
    pub tools: Vec<Box<dyn AiTool>>,
    provider: String,
}

const AKASH_MODELS: [&str; 2] = ["Qwen3-235B-A22B-FP8", "DeepSeek-R1-0528"];

impl LLMService {
    pub fn new(
        system_prompt: Option<&str>,
        tools: Vec<Box<dyn AiTool>>,
        provider: &str,
    ) -> Result<Self, Error> {
        let akash_resolver = ServiceTargetResolver::from_resolver_fn(
            |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                let ServiceTarget { ref model, .. } = service_target;
                let model_name = model.model_name.to_string();
                if AKASH_MODELS.contains(&model_name.as_str()) {
                    let endpoint = Endpoint::from_static("https://chatapi.akash.network/api/v1/");
                    let auth = AuthData::from_env("AKASH_API_KEY");
                    let model = ModelIden::new(AdapterKind::OpenAI, model_name);
                    Ok(ServiceTarget {
                        endpoint,
                        auth,
                        model,
                    })
                } else {
                    Ok(service_target)
                }
            },
        );

        // Compose tool context for the system prompt
        let mut tool_context = String::new();
        if !tools.is_empty() {
            tool_context.push_str("You have access to the following tools:\n");
            for tool in &tools {
                tool_context.push_str(&format!("- {}: {}\n", tool.name(), tool.description()));
            }
            tool_context.push_str(
                "For each function call, follow this exact format:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name\n```json\n{\"param1\": \"value1\", \"param2\": \"value2\"}\n```\n<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n",
            );
        }

        let merged_prompt = match system_prompt {
            Some(user_prompt) if !user_prompt.trim().is_empty() => {
                format!("{}\n\n{}", tool_context, user_prompt)
            }
            _ => tool_context,
        };

        let client = Client::builder()
            .with_service_target_resolver(akash_resolver)
            .build();

        Ok(LLMService {
            provider: provider.to_string(),
            client,
            system_prompt: Some(merged_prompt),
            tools,
        })
    }

    pub fn add_tool(&mut self, tool: Box<dyn AiTool>) {
        self.tools.push(tool);
    }

    pub fn remove_tool(&mut self, tool_name: &str) -> Result<(), Error> {
        if let Some(pos) = self.tools.iter().position(|t| t.name() == tool_name) {
            self.tools.remove(pos);
            Ok(())
        } else {
            Err(anyhow!("Tool not found: {}", tool_name))
        }
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
    }

    pub fn list_tools(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
    }

    pub fn find_tool(&self, tool_name: &str) -> Option<&Box<dyn AiTool>> {
        self.tools.iter().find(|t| t.name() == tool_name)
    }
}

#[async_trait]
impl AiService for LLMService {
    async fn generate_response(
        &self,
        messages: &Vec<ChatMessage>,
        searched_messages: Option<&Vec<ChatMessage>>,
    ) -> Result<String, Error> {
        let mut all_msgs = vec![ChatMessage::system(
            self.system_prompt.clone().unwrap_or_default(),
        )];

        if let Some(searched_msgs) = searched_messages {
            all_msgs.push(ChatMessage::system(
                "The following messages may help you when responding to the user. You can use them, or not.",
            ));
            all_msgs.extend(searched_msgs.to_owned());
            all_msgs.push(ChatMessage::system(
                "End of search results. Following are messages from the conversation thread history.",
            ));
        }

        all_msgs.extend(messages.to_owned());
        let chat_req = ChatRequest::new(all_msgs);

        let chat_response = self
            .client
            .exec_chat(&self.provider, chat_req, None)
            .await?;

        if let Some(content) = chat_response.content {
            let response_text = content
                .text_into_string()
                .ok_or(anyhow!("No content in AI response from AkashChatService"))?;

            return Ok(response_text);
        } else {
            Err(anyhow!("No content in AI response from AkashChatService"))
        }
    }

    async fn generate_response_stream<'a>(
        &'a self,
        messages: &'a Vec<ChatMessage>,
        searched_messages: Option<&'a Vec<ChatMessage>>,
    ) -> Result<
        Pin<
            Box<
                dyn futures_core::Stream<Item = Result<genai::chat::ChatStreamEvent, anyhow::Error>>
                    + Send
                    + 'a,
            >,
        >,
        anyhow::Error,
    > {
        let mut all_msgs = vec![ChatMessage::system(
            self.system_prompt.clone().unwrap_or_default(),
        )];

        if let Some(searched_msgs) = searched_messages {
            all_msgs.push(ChatMessage::system(
                "The following messages may help you when responding to the user. You can use them, or not.",
            ));
            all_msgs.extend(searched_msgs.to_owned());
            all_msgs.push(ChatMessage::system(
                "End of search results. Following are messages from the conversation thread history.",
            ));
        }

        all_msgs.extend(messages.to_owned());
        let chat_req = ChatRequest::new(all_msgs);

        let chat_stream_response = self
            .client
            .exec_chat_stream(&self.provider, chat_req, None)
            .await?;

        let mapped_stream = chat_stream_response
            .stream
            .map(|event_result| event_result.map_err(anyhow::Error::from));
        Ok(Box::pin(mapped_stream))
    }
}
