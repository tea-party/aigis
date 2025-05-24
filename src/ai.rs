use anyhow::{anyhow, Error};
use async_trait::async_trait;
use genai::chat::{ChatMessage, ChatRequest, ChatRole};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{AdapterKind, Client, ClientBuilder, Credentials, ModelIden, ServiceTarget};
use std::env;

// Define the AiService trait
#[async_trait]
pub trait AiService {
    async fn generate_response(&self, prompt: &str) -> Result<String, Error>;
}

// Structure for AkashChatService
pub struct AkashChatService {
    client: Client,
    system_prompt: Option<String>,
}

impl AkashChatService {
    pub fn new(system_prompt: Option<String>) -> Result<Self, Error> {
        let api_key = env::var("AKASH_API_KEY")
            .map_err(|_| anyhow!("AKASH_API_KEY not found in environment"))?;

        let resolver = ServiceTargetResolver::new(
            ServiceTarget::new(Endpoint::new("https://chatapi.akash.network/api/v1/")),
            AuthData::from_bearer(api_key),
            ModelIden::new(AdapterKind::OpenAI, "gpt-3.5-turbo"), // Default model for Akash
        );

        let client = ClientBuilder::new()
            .resolver(resolver)
            .build()
            .map_err(|e| anyhow!("Failed to build AkashChatService client: {}", e))?;

        Ok(AkashChatService {
            client,
            system_prompt,
        })
    }
}

#[async_trait]
impl AiService for AkashChatService {
    async fn generate_response(&self, prompt: &str) -> Result<String, Error> {
        let mut messages = Vec::new();

        if let Some(sys_prompt) = &self.system_prompt {
            messages.push(ChatMessage::new(ChatRole::System, sys_prompt.clone()));
        }
        messages.push(ChatMessage::new(ChatRole::User, prompt.to_string()));

        let chat_req = ChatRequest::new(messages);
        
        let model_to_use = ModelIden::new(AdapterKind::OpenAI, "gpt-3.5-turbo");
        let chat_response = self.client.exec_chat(&model_to_use, chat_req).await?;

        if let Some(content) = chat_response.content {
            Ok(content)
        } else if let Some(content_parts) = chat_response.content_parts {
             Ok(content_parts.iter().map(|p| p.text()).collect::<Vec<_>>().join(""))
        }
        else {
            Err(anyhow!("No content in AI response from AkashChatService"))
        }
    }
}

// Structure for GeminiService
pub struct GeminiService {
    client: Client,
    system_prompt: Option<String>,
}

impl GeminiService {
    pub fn new(system_prompt: Option<String>) -> Result<Self, Error> {
        let api_key = env::var("GEMINI_API_KEY")
            .map_err(|_| anyhow!("GEMINI_API_KEY not found in environment"))?;

        let credentials = Credentials::new(Some(api_key), None, None, None)
            .map_err(|e| anyhow!("Failed to create credentials for GeminiService: {}", e))?;

        let client = ClientBuilder::new()
            .credentials(credentials)
            .build()
            .map_err(|e| anyhow!("Failed to build GeminiService client: {}", e))?;

        Ok(GeminiService {
            client,
            system_prompt,
        })
    }
}

#[async_trait]
impl AiService for GeminiService {
    async fn generate_response(&self, prompt: &str) -> Result<String, Error> {
        let mut messages = Vec::new();

        if let Some(sys_prompt) = &self.system_prompt {
            let combined_prompt = format!("System Instructions: {}\n\nUser Prompt: {}", sys_prompt, prompt);
            messages.push(ChatMessage::new(ChatRole::User, combined_prompt));
        } else {
            messages.push(ChatMessage::new(ChatRole::User, prompt.to_string()));
        }
        
        let chat_req = ChatRequest::new(messages);

        let model = ModelIden::new(AdapterKind::Google, "gemini-pro");
        let chat_response = self.client.exec_chat(&model, chat_req).await?;

        if let Some(content) = chat_response.content {
            Ok(content)
        } else if let Some(content_parts) = chat_response.content_parts {
             Ok(content_parts.iter().map(|p| p.text()).collect::<Vec<_>>().join(""))
        }
        else {
            Err(anyhow!("No content in AI response from GeminiService"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // Helper function to temporarily set an environment variable
    fn with_env_var<F>(key: &str, value: Option<&str>, mut f: F)
    where
        F: FnMut(),
    {
        let original_value = env::var(key).ok();
        if let Some(v) = value {
            env::set_var(key, v);
        } else {
            env::remove_var(key);
        }
        
        // Use a try/catch block to ensure environment variable is restored even if test panics
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut f));
        
        if let Some(orig_v) = original_value {
            env::set_var(key, orig_v);
        } else {
            env::remove_var(key);
        }
        
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn akash_chat_service_new_missing_key_fails() {
        with_env_var("AKASH_API_KEY", None, || {
            let result = AkashChatService::new(None);
            assert!(result.is_err(), "Expected error when AKASH_API_KEY is not set");
            if let Err(e) = result {
                assert!(e.to_string().contains("AKASH_API_KEY not found"), "Error message mismatch: {}", e);
            }
        });
    }

    #[test]
    fn akash_chat_service_new_with_key_succeeds() {
        with_env_var("AKASH_API_KEY", Some("dummy_key"), || {
            let result = AkashChatService::new(None);
            // This test primarily checks that `new` doesn't panic or return an env-related error.
            // It might still fail if the dummy_key is invalid for other reasons during client setup,
            // but it shouldn't fail due to the key *not being found*.
            // The current implementation of AkashChatService::new only checks for key presence.
            assert!(result.is_ok(), "Expected Ok when AKASH_API_KEY is set, got: {:?}", result.err());
        });
    }

    #[test]
    fn gemini_service_new_missing_key_fails() {
        with_env_var("GEMINI_API_KEY", None, || {
            let result = GeminiService::new(None);
            assert!(result.is_err(), "Expected error when GEMINI_API_KEY is not set");
            if let Err(e) = result {
                assert!(e.to_string().contains("GEMINI_API_KEY not found"), "Error message mismatch: {}", e);
            }
        });
    }

    #[test]
    fn gemini_service_new_with_key_succeeds() {
        with_env_var("GEMINI_API_KEY", Some("dummy_key"), || {
            let result = GeminiService::new(None);
            // Similar to Akash, this checks successful construction if key is present.
            assert!(result.is_ok(), "Expected Ok when GEMINI_API_KEY is set, got: {:?}", result.err());
        });
    }
}
