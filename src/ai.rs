use anyhow::{Error, anyhow};
use async_trait::async_trait;
use genai::chat::{ChatMessage, ChatRequest};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ModelIden, ServiceTarget, adapter::AdapterKind};

#[async_trait]
pub trait AiService {
    async fn generate_response(&self, messages: &Vec<ChatMessage>) -> Result<String, Error>;
}

// Structure for AkashChatService
pub struct LLMService {
    client: Client,
    system_prompt: Option<String>,
}

const AKASH_MODELS: [&str; 1] = ["Qwen3-235B-A22B-FP8"];

impl LLMService {
    pub fn new(system_prompt: Option<&str>) -> Result<Self, Error> {
        let akash_resolver = ServiceTargetResolver::from_resolver_fn(
            |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                let ServiceTarget { ref model, .. } = service_target;
                // if we have a model supported by Akash
                if AKASH_MODELS.contains(&&model.model_name.to_string().as_str()) {
                    let endpoint = Endpoint::from_static("https://chatapi.akash.network/api/v1/");
                    let auth = AuthData::from_env("AKASH_API_KEY");
                    let model = ModelIden::new(AdapterKind::OpenAI, "Qwen3-235B-A22B-FP8");
                    Ok(ServiceTarget {
                        endpoint,
                        auth,
                        model,
                    })
                } else {
                    // default to the included resolver
                    Ok(service_target)
                }
            },
        );

        // -- Build the new client with this adapter_config
        let client = Client::builder()
            .with_service_target_resolver(akash_resolver)
            .build();

        Ok(LLMService {
            client,
            system_prompt: system_prompt.map(|s| s.to_string()),
        })
    }
}

#[async_trait]
impl AiService for LLMService {
    async fn generate_response(&self, messages: &Vec<ChatMessage>) -> Result<String, Error> {
        // add our prompt to the beginning of the messages
        let mut all_msgs = vec![ChatMessage::system(
            self.system_prompt.clone().unwrap_or_default(),
        )];
        all_msgs.extend(messages.to_owned());
        let chat_req = ChatRequest::new(all_msgs);

        let chat_response = self
            .client
            .exec_chat("Qwen3-235B-A22B-FP8", chat_req, None)
            .await?;

        if let Some(content) = chat_response.content {
            return content
                .text_into_string()
                .ok_or(anyhow!("No content in AI response from AkashChatService"));
        } else {
            Err(anyhow!("No content in AI response from AkashChatService"))
        }
    }
}
