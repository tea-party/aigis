use anyhow::{anyhow, Context, Error}; // Added Context
use async_trait::async_trait;
use genai::chat::{ChatMessage, ChatRequest, ChatRole};
// Updated genai imports
use genai::{
    AdapterKind, Client, ClientBuilder, ModelIden,
    resolver::{AuthData, Endpoint, Error as ResolverError, /* ModelIden as ResolverModelIden (if needed) */ ServiceTarget, ServiceTargetResolver},
};
use std::sync::Arc;
use std::env; // Added std::env

// Define the AiService trait
#[async_trait]
pub trait AiService {
    async fn generate_response(&self, prompt: &str) -> Result<String, Error>;
}

// Structure for AkashChatService
pub struct AkashChatService {
    client: Arc<Client>,
    system_prompt: Option<String>,
}

impl AkashChatService {
    const AKASH_CHAT_MODEL_NAME: &'static str = "akash/gpt-3.5-turbo";

    pub fn new(client: Arc<Client>, system_prompt: Option<String>) -> Self {
        Self {
            client,
            system_prompt,
        }
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
        
        let model_to_use = Self::AKASH_CHAT_MODEL_NAME; // Pass the string slice directly
        let chat_response = self.client.exec_chat(model_to_use, chat_req).await?;

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
    client: Arc<Client>,
    system_prompt: Option<String>,
}

impl GeminiService {
    const GEMINI_MODEL_NAME: &'static str = "gemini/gemini-pro";

    pub fn new(client: Arc<Client>, system_prompt: Option<String>) -> Self {
        Self {
            client,
            system_prompt,
        }
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

        let model_to_use = Self::GEMINI_MODEL_NAME; // Pass the string slice directly
        let chat_response = self.client.exec_chat(model_to_use, chat_req).await?;

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

pub fn create_custom_resolver() -> ServiceTargetResolver {
    ServiceTargetResolver::from_resolver_fn(|mut service_target: ServiceTarget| -> Result<ServiceTarget, ResolverError> {
        let model_name_full = service_target.model_name().to_string();

        if model_name_full.starts_with("akash/") {
            let akash_api_key = env::var("AKASH_API_KEY")
                .map_err(|e| ResolverError::Auth(format!("AKASH_API_KEY not found: {}", e)))?;
            
            let endpoint = Endpoint::try_from("https://chatapi.akash.network/api/v1/")
                .map_err(|e| ResolverError::Config(format!("Invalid Akash endpoint: {}", e)))?;
            
            let auth = AuthData::from_bearer(akash_api_key);
            
            let model_name_stripped = model_name_full.strip_prefix("akash/").unwrap_or_default();
            let model = ModelIden::new(AdapterKind::OpenAI, model_name_stripped);
            
            // Update the existing service_target
            service_target.endpoint = Some(endpoint);
            service_target.auth = Some(auth);
            service_target.model = model; // ModelIden is not Option in ServiceTarget
            
            Ok(service_target)
        } else if model_name_full.starts_with("gemini/") {
            let gemini_api_key = env::var("GEMINI_API_KEY")
                .map_err(|e| ResolverError::Auth(format!("GEMINI_API_KEY not found: {}", e)))?;

            let auth = AuthData::from_api_key(gemini_api_key);

            let model_name_stripped = model_name_full.strip_prefix("gemini/").unwrap_or_default();
            let model = ModelIden::new(AdapterKind::GoogleAi, model_name_stripped); 
                            
            // Update the existing service_target
            service_target.auth = Some(auth);
            service_target.model = model;
            // For Gemini, if genai handles the endpoint by default for GoogleAi, we might not need to set it.
            // If a specific endpoint is needed, it should be set here.
            // service_target.endpoint = Some(Endpoint::try_from("https://generativelanguage.googleapis.com/v1beta").unwrap());


            Ok(service_target)
        } else {
            // Default: pass through if no prefix matches
            Ok(service_target)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*; // Imports AiService, AkashChatService, GeminiService, create_custom_resolver
    use std::env;
    use std::sync::Arc;
    use genai::Client; // For dummy client
    // For resolver tests: ServiceTarget, AuthData, ModelIden, AdapterKind are already in scope via super::* from the main genai import block.
    // Explicitly importing them again like `use genai::resolver::{ServiceTarget, AuthData, ModelIden};`
    // and `use genai::AdapterKind;` is not strictly necessary if they are part of the `genai::{...}` block above.
    // However, it can make it clearer where these types are coming from for the tests.
    // For this version, I'll rely on them being in scope from `super::*` and the main file's `use genai::{...}`.

    // Simplified env var helper for resolver tests
    fn with_env_vars<F>(vars: Vec<(&str, Option<&str>)>, mut f: F)
    where
        F: FnMut(),
    {
        let mut original_vars = Vec::new();
        for (key, value) in &vars {
            original_vars.push((*key, env::var(*key).ok()));
            if let Some(v) = value {
                env::set_var(*key, v);
            } else {
                env::remove_var(*key);
            }
        }
        
        // Use a try/catch block to ensure environment variables are restored even if test panics
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut f));
        
        for (key, original_value) in original_vars {
            if let Some(orig_v) = original_value {
                env::set_var(key, orig_v);
            } else {
                env::remove_var(key);
            }
        }

        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn can_create_services_with_dummy_client() {
        // Create a basic client (won't be functional for actual calls without a proper resolver)
        let client = Client::builder().build().expect("Failed to build dummy client");
        let shared_client = Arc::new(client);

        let _akash_service = AkashChatService::new(shared_client.clone(), None);
        // Simple assertion to ensure construction doesn't panic and const is accessible
        assert_eq!(AkashChatService::AKASH_CHAT_MODEL_NAME, "akash/gpt-3.5-turbo"); 

        let _gemini_service = GeminiService::new(shared_client.clone(), None);
        assert_eq!(GeminiService::GEMINI_MODEL_NAME, "gemini/gemini-pro");
    }

    #[test]
    fn test_custom_resolver_akash_routing_ok() {
        with_env_vars(vec![("AKASH_API_KEY", Some("test_akash_key"))], || {
            let resolver = create_custom_resolver();
            let st = ServiceTarget::new("akash/gpt-3.5-turbo"); // ServiceTarget can take &str
            let resolved_st = resolver.resolve(st).expect("Resolver failed for Akash");

            assert_eq!(resolved_st.endpoint.as_ref().unwrap().base().to_string(), "https://chatapi.akash.network/api/v1/");
            assert_eq!(resolved_st.model.adapter_kind(), AdapterKind::OpenAI);
            assert_eq!(resolved_st.model.model_name(), "gpt-3.5-turbo");
            assert!(matches!(resolved_st.auth.as_ref().unwrap(), AuthData::Bearer(_)), "Akash auth not bearer");
        });
    }

    #[test]
    fn test_custom_resolver_gemini_routing_ok() {
        with_env_vars(vec![("GEMINI_API_KEY", Some("test_gemini_key"))], || {
            let resolver = create_custom_resolver();
            let st = ServiceTarget::new("gemini/gemini-pro");
            let resolved_st = resolver.resolve(st).expect("Resolver failed for Gemini");
            
            assert_eq!(resolved_st.model.adapter_kind(), AdapterKind::GoogleAi);
            assert_eq!(resolved_st.model.model_name(), "gemini-pro");
            assert!(matches!(resolved_st.auth.as_ref().unwrap(), AuthData::ApiKey(_)), "Gemini auth not ApiKey");
            // Default endpoint for GoogleAI is usually handled by genai, so not asserting endpoint.
        });
    }

    #[test]
    fn test_custom_resolver_default_passthrough() {
        with_env_vars(vec![("AKASH_API_KEY", None), ("GEMINI_API_KEY", None)], || {
            let resolver = create_custom_resolver();
            // Create an initial ServiceTarget. Its auth and endpoint might be None by default.
            let original_st_model_name = "some-other-model/variant";
            let original_st = ServiceTarget::new(original_st_model_name);
            
            let resolved_st = resolver.resolve(original_st).expect("Resolver failed for passthrough");
            
            assert_eq!(resolved_st.model.model_name(), original_st_model_name);
            // Check that auth and endpoint are still in their default (None) state
            assert!(resolved_st.auth.is_none(), "Auth should be None for passthrough if default is None");
            assert!(resolved_st.endpoint.is_none(), "Endpoint should be None for passthrough if default is None");
        });
    }

    #[test]
    fn test_custom_resolver_akash_missing_key() {
        with_env_vars(vec![("AKASH_API_KEY", None)], || { // Ensure key is unset
            let resolver = create_custom_resolver();
            let st = ServiceTarget::new("akash/gpt-3.5-turbo");
            let result = resolver.resolve(st);
            assert!(result.is_err(), "Expected error for Akash with missing key");
            if let Err(e) = result {
                assert!(e.to_string().contains("AKASH_API_KEY not found"), "Error message mismatch: {}", e);
            }
        });
    }

    #[test]
    fn test_custom_resolver_gemini_missing_key() {
        with_env_vars(vec![("GEMINI_API_KEY", None)], || { // Ensure key is unset
            let resolver = create_custom_resolver();
            let st = ServiceTarget::new("gemini/gemini-pro");
            let result = resolver.resolve(st);
            assert!(result.is_err(), "Expected error for Gemini with missing key");
            if let Err(e) = result {
                assert!(e.to_string().contains("GEMINI_API_KEY not found"), "Error message mismatch: {}", e);
            }
        });
    }
}
