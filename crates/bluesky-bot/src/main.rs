use std::{
    collections::HashMap,
    option::Option,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Result;
use async_trait::async_trait;
use atrium_api::{
    app::bsky::feed::{
        defs::{PostViewData, ThreadViewPostData},
        get_post_thread,
        post::ReplyRefData,
    },
    com::atproto::repo::strong_ref::MainData,
    types::{
        string::{Cid, Datetime, Did, Language},
        LimitedU16, Object, TryFromUnknown,
    },
};
use bsky_sdk::BskyAgent;
use genai::chat::ChatMessage;
use logi::embed::Embedder;
use logi::llm::{AiService, LLMService};
use logi::tools::AiTool;
use logi::{
    cursor::{self, load_cursor},
    vdb::MemoryEntry,
};
use metrics_exporter_prometheus::PrometheusBuilder;
use once_cell::sync::Lazy;

use multibase::Base;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, trace};

// Import logi tools directly
use genai::chat::ToolResponse;
use logi::tools::calc::MathTool;
use logi::tools::search::DDGSearchTool;
use logi::tools::website::WebsiteTool;
use logi::tools::{execute_tool_calls, parse_tool_calls};

use rocketman::{
    connection::JetstreamConnection,
    handler,
    ingestion::LexiconIngestor,
    options::JetstreamOptions,
    types::event::{Commit, Event},
};

use logi::vdb::MemoryStore;
use std::time::{SystemTime, UNIX_EPOCH};

static POSTS_INGESTED: Lazy<metrics::Counter> =
    Lazy::new(|| metrics::counter!("posts_ingested_total"));
static INGEST_ERRORS: Lazy<metrics::Counter> =
    Lazy::new(|| metrics::counter!("ingest_errors_total"));
static INGEST_LATENCY: Lazy<metrics::Histogram> =
    Lazy::new(|| metrics::histogram!("ingest_latency_seconds"));
static TOOL_CALL_TIMES: u8 = 3; // Maximum number of identical tool calls before breaking loop

fn setup_metrics() {
    // Initialize metrics here
    if let Err(e) = PrometheusBuilder::new().install() {
        error!(
            "Failed to install, program will run without Prometheus exporter: {}",
            e
        );
    }
}

async fn setup_bsky_sess() -> anyhow::Result<(BskyAgent, Did)> {
    let span = tracing::info_span!("setup_bsky_sess");
    let _enter = span.enter();

    let agent = BskyAgent::builder().build().await?;
    let res = agent
        .login(std::env::var("ATP_USER")?, std::env::var("ATP_PASSWORD")?)
        .await?;

    info!("logged in as {}", res.handle.to_string());

    Ok((agent, res.did.to_owned()))
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();
    setup_metrics();
    println!("initialising gorkai v0.1.0");

    let main_span = tracing::info_span!("main");
    let _main_enter = main_span.enter();

    info!("gorkin it...");

    let (agent, did) = match setup_bsky_sess().await {
        Ok(r) => r,
        Err(e) => panic!("{}", e.to_string()),
    };
    // init the builder
    let opts = JetstreamOptions::builder()
        // your EXACT nsids
        .wanted_collections(vec!["app.bsky.feed.post".to_string()])
        .build();
    // create the jetstream connector
    let jetstream = JetstreamConnection::new(opts);

    // create your ingestors
    let mut ingestors: HashMap<String, Box<dyn LexiconIngestor + Send + Sync>> = HashMap::new();

    // Read the ALLOWED_USERS environment variable and parse it
    let allowlist = match std::env::var("ALLOWED_USERS") {
        Ok(users_str) if !users_str.trim().is_empty() => {
            let users: Vec<String> = users_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if users.is_empty() {
                None
            } else {
                Some(users)
            }
        }
        _ => None,
    };

    // get the time
    let current_time = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap();

    let system_message = match std::fs::read_to_string("./prompt.txt") {
        Ok(content) => Some(content + &format!("\n\nCurrent time: {}", current_time)),
        Err(e) => {
            error!("Could not read ./prompt.txt: {}", e);
            info!("Using default system message for AI service.");
            None
        }
    };

    info!("Initializing AI service with tools...");

    let vdb = MemoryStore::new(
        &std::env::var("QDRANT_URL").expect("qdrant url not set"),
        &std::env::var("QDRANT_DB").unwrap_or("aigis-db".to_string()),
        1536, // embedding dimension, change if needed
    )
    .await
    .expect("qdrant db failed initialization");

    // Note: Tools are initialized in PostListener::new

    ingestors.insert(
        // your EXACT nsid
        "app.bsky.feed.post".to_string(),
        Box::new(PostListener::new(
            agent.clone(),
            did,
            vdb,
            allowlist,
            system_message,
        )),
    );

    let worker_count: usize = std::env::var("WORKER_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3); // Default to 3 workers if not set

    let semaphore = Arc::new(Semaphore::new(worker_count));

    // tracks the last message we've processed
    let cursor: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(load_cursor().await));

    // get channels
    let msg_rx = jetstream.get_msg_rx();
    let reconnect_tx = jetstream.get_reconnect_tx();

    // spawn a task to process messages from the queue.
    // this is a simple implementation, you can use a more complex one based on needs.
    let c_cursor = cursor.clone();
    let c_semaphore = semaphore.clone();
    // Clone ingestors to move into the task.
    let arcgestors = Arc::new(ingestors);
    tokio::spawn(async move {
        while let Ok(message) = msg_rx.recv_async().await {
            // Clone semaphore permit for the spawned task
            let permit = c_semaphore.clone().acquire_owned().await.unwrap();
            // Clone necessary variables for the spawned task
            let ingestors = arcgestors.clone();
            let reconnect_tx = reconnect_tx.clone();
            let c_cursor = c_cursor.clone();

            tokio::spawn(async move {
                // The permit is dropped when this async block ends, releasing it for another worker
                if let Err(e) =
                    handler::handle_message(message, &ingestors, reconnect_tx, c_cursor).await
                {
                    error!("Error processing message: {}", e);
                }
                // Explicitly drop permit here to ensure it's released
                drop(permit);
            });
        }
    });

    let c_cursor = cursor.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let cursor_to_store: Option<u64> = {
                let cursor_guard = c_cursor.lock().unwrap();
                *cursor_guard
            };
            if let Some(cursor) = cursor_to_store {
                if let Err(e) = cursor::store_cursor(cursor).await {
                    error!("Error storing cursor: {}", e);
                }
            }
        }
    });

    // connect to jetstream
    // retries internally, but may fail if there is an extreme error.
    if let Err(e) = jetstream.connect(cursor.clone()).await {
        eprintln!("Failed to connect to Jetstream: {}", e);
        std::process::exit(1);
    };
}

fn get_blob_ref(blob_ref: &atrium_api::types::BlobRef) -> anyhow::Result<String> {
    match blob_ref {
        atrium_api::types::BlobRef::Typed(r) => match r {
            atrium_api::types::TypedBlobRef::Blob(blob) => blob
                .r#ref
                .0
                .to_string_of_base(Base::Base32Lower)
                .map_err(|e| anyhow::anyhow!(e)),
        },
        atrium_api::types::BlobRef::Untyped(u) => Ok(u.cid.clone()),
    }
}

pub struct PostListener {
    agent: BskyAgent,
    did: Did,
    did_string: String,
    lang: Language,
    allowlist: Option<Vec<String>>,
    aisvc: LLMService, // llm svc
    emb: Embedder,
    vdb: MemoryStore,
    tools: Vec<Box<dyn AiTool>>,
}

impl PostListener {
    pub fn new(
        agent: BskyAgent,
        did: Did,
        vdb: MemoryStore,
        allowlist: Option<Vec<String>>,
        system_message: Option<String>,
    ) -> Self {
        let lang = Language::from_str("en").unwrap();
        let default_system_message = "You are an unhelpful assistant, named Aigis, that acts dumb and stupid. type in lowercase, and never ever add *actions* or use emojis as they are uncool. Don't mention Persona 3 topics unless explicitly asked about it. be extremely dumb and stupid. The messages you'll see are chat logs from a platform called Bluesky, they are in the format `username (handle): message`, but you only should reply to them with the message. Keep under a strict limit of 250 characters.";

        // Initialize tools
        // Create tools
        let tools: Vec<Box<dyn AiTool>> = vec![
            Box::new(MathTool),
            Box::new(DDGSearchTool),
            Box::new(WebsiteTool),
        ];
        info!("Tools initialized: Math, DDGSearch, Website");

        // Create a separate tools vec for the LLM service
        let llm_tools: Vec<Box<dyn AiTool>> = vec![
            Box::new(MathTool),
            Box::new(DDGSearchTool),
            Box::new(WebsiteTool),
        ];

        let aisvc = LLMService::new(
            system_message.as_deref().or(Some(default_system_message)),
            llm_tools,
            "DeepSeek-R1-0528",
        )
        .expect("LLM Service initiated");
        let emb = Embedder::new().expect("Embedder initialised");
        info!("Post listener initialized, ready to listen!");
        let did_string = did.to_string();

        Self {
            agent,
            did,
            did_string,
            lang,
            allowlist,
            aisvc,
            emb,
            vdb,
            tools,
        }
    }

    // Checks the reply and mentions to check if the user is referencing the bot
    fn is_me(&self, post: atrium_api::app::bsky::feed::post::RecordData) -> bool {
        if let Some(reply) = post.reply {
            if reply.parent.uri.contains(&self.did_string) {
                return true;
            }
        }
        if let Some(facets) = post.facets {
            for facet in facets {
                for ftr in facet.data.features {
                    match ftr {
                        atrium_api::types::Union::Refs(e) => match e {
                            atrium_api::app::bsky::richtext::facet::MainFeaturesItem::Mention(
                                object,
                            ) => {
                                if object.did == self.did {
                                    return true;
                                } else {
                                    continue;
                                }
                            }
                            _ => continue,
                        },
                        atrium_api::types::Union::Unknown(_) => continue,
                    }
                }
            }
        }
        // is the quoted post us?
        if let Some(embed) = post.embed {
            match embed {
                atrium_api::types::Union::Refs(e) => match e {
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedRecordMain(object) => {
                        if object.record.uri.contains(&self.did_string) {
                            return true;
                        }
                    },
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedRecordWithMediaMain(object) =>{
                        if object.record.data.record.data.uri.contains(&self.did_string){
                            return true;
                        }
                    },
                    _ => (),
                },
                _ => ()
            }
        }
        false
    }

    fn is_allowlisted(&self, did: &str) -> bool {
        if let Some(ref allowlist) = self.allowlist {
            allowlist.iter().any(|x| x == did)
        } else {
            true
        }
    }

    fn build_reply_ref(
        &self,
        reply: Option<Object<ReplyRefData>>,
        rcid: Cid,
        msg_did: String,
        collection: String,
        rkey: String,
    ) -> Object<ReplyRefData> {
        if let Some(mut reply) = reply {
            reply.parent = MainData {
                cid: rcid,
                uri: format!("at://{}/{}/{}", msg_did, collection, rkey),
            }
            .into();
            reply
        } else {
            ReplyRefData {
                parent: MainData {
                    cid: rcid.clone(),
                    uri: format!("at://{}/{}/{}", msg_did, collection, rkey),
                }
                .into(),
                root: MainData {
                    cid: rcid,
                    uri: format!("at://{}/{}/{}", msg_did, collection, rkey),
                }
                .into(),
            }
            .into()
        }
    }

    pub fn assemble_post_message(
        &self,
        post: Object<atrium_api::app::bsky::feed::defs::PostViewData>,
    ) -> Result<String> {
        let author = post
            .author
            .display_name
            .clone()
            .map(|e| format!("{} ({})", e, post.author.handle.as_str()))
            .unwrap_or_else(|| post.author.handle.as_str().to_owned());
        let record_data =
            atrium_api::app::bsky::feed::post::RecordData::try_from_unknown(post.record.clone())?;

        Ok(format!("{}: {}", author, record_data.text))
    }

    /// Extracts basic post data for JSON serialization from a BlueSky post
    ///
    /// This function converts a BlueSky post object into our simplified PostData
    /// structure that can be easily serialized to JSON.
    pub fn extract_post_data(
        &self,
        post: Object<atrium_api::app::bsky::feed::defs::PostViewData>,
    ) -> Result<PostData> {
        let author = post
            .author
            .display_name
            .clone()
            .map(|e| format!("{} ({})", e, post.author.handle.as_str()))
            .unwrap_or_else(|| post.author.handle.as_str().to_owned());
        let record_data =
            atrium_api::app::bsky::feed::post::RecordData::try_from_unknown(post.record.clone())?;

        // Extract embed data if present
        let embed = self.extract_post_embed(&record_data);

        Ok(PostData {
            author,
            text: record_data.text,
            uri: post.uri.to_string(),
            author_did: post.author.did.to_string(),
            indexed_at: Some(post.indexed_at.as_str().to_owned()),
            embed,
        })
    }

    /// Extract embed data from a BlueSky post
    fn extract_post_embed(
        &self,
        record_data: &atrium_api::app::bsky::feed::post::RecordData,
    ) -> Option<PostEmbed> {
        if let Some(embed) = &record_data.embed {
            match embed {
                atrium_api::types::Union::Refs(e) => match e {
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedImagesMain(object) => {
                        let images = object
                            .images
                            .iter()
                            .map(|img| PostEmbedImage {
                                image: get_blob_ref(&img.image).unwrap_or("".to_string()),
                                alt: Some(img.alt.clone()),
                            })
                            .collect();

                        Some(PostEmbed::Images(PostEmbedImages { images }))
                    },
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedVideoMain(object) => {
                        Some(PostEmbed::Video(PostEmbedVideo {
                            video: get_blob_ref(&object.video).unwrap_or("".to_string()),
                            duration: None, // API doesn't provide duration directly
                        }))
                    },
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedExternalMain(object) => {
                        Some(PostEmbed::External(PostEmbedExternal {
                            uri: object.external.uri.to_string(),
                            title: Some(object.external.title.clone()),
                            description: Some(object.external.description.clone()),
                        }))
                    },
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedRecordMain(object) => {
                        Some(PostEmbed::Record(PostEmbedRecord {
                            record: object.record.uri.to_string(),
                            title: None, // Record embeds don't have titles in the API
                        }))
                    },
                    atrium_api::app::bsky::feed::post::RecordEmbedRefs::AppBskyEmbedRecordWithMediaMain(object) => {
                        let record = PostEmbedRecord {
                            record: object.record.record.uri.to_string(),
                            title: None,
                        };

                        let media = match &object.media {
                            atrium_api::types::Union::Refs(media_ref) => match media_ref {
                                atrium_api::app::bsky::embed::record_with_media::MainMediaRefs::AppBskyEmbedImagesMain(images_obj) => {
                                    let images = images_obj
                                        .images
                                        .iter()
                                        .map(|img| PostEmbedImage {
                                            image: get_blob_ref(&img.image).unwrap_or("".to_string()),
                                            alt: Some(img.alt.clone()),
                                        })
                                        .collect();

                                    vec![PostEmbedMedia::Images(PostEmbedImages { images })]
                                },
                                atrium_api::app::bsky::embed::record_with_media::MainMediaRefs::AppBskyEmbedVideoMain(video_obj) => {
                                    vec![PostEmbedMedia::Video(PostEmbedVideo {
                                        video: get_blob_ref(&video_obj.video).unwrap_or("".to_string()),
                                        duration: None, // API doesn't provide duration directly
                                    })]
                                },
                                _ => vec![],
                            },
                            _ => vec![],
                        };

                        Some(PostEmbed::RecordWithMedia(PostEmbedRecordWithMedia {
                            record,
                            media,
                        }))
                    },
                },
                atrium_api::types::Union::Unknown(_) => None,
            }
        } else {
            None
        }
    }

    // New helper function to collect posts by traversing up the parent chain
    fn collect_parents_recursive(
        &self,
        current_thread_view: Box<Object<ThreadViewPostData>>,
        posts: &mut Vec<Object<PostViewData>>,
    ) -> Result<()> {
        // Add the current post to the list
        posts.push(current_thread_view.post.clone());

        // Check if there's a parent and if it's a ThreadViewPost
        if let Some(parent_thread_view) = current_thread_view.parent.clone() {
            match parent_thread_view {
                atrium_api::types::Union::Refs(r) => match r {
                    atrium_api::app::bsky::feed::defs::ThreadViewPostParentRefs::ThreadViewPost(
                        parent_object,
                    ) => {
                        // Recursively call for the parent
                        self.collect_parents_recursive(parent_object, posts)?;
                    }
                    // Stop if the parent is a ThreadViewNotFound or other type
                    _ => {}
                },
                // Stop if the parent is an unknown type
                atrium_api::types::Union::Unknown(_) => {}
            };
        }
        Ok(())
    }

    /// Extracts a thread as a collection of structured JSON-serializable PostData objects
    ///
    /// This function fetches a thread by its URI and returns a vector of PostData objects
    /// representing each post in the thread in chronological order (oldest to newest).
    pub async fn atp_thread_to_json(&self, uri: &str) -> Result<Vec<PostData>> {
        let mut all_posts: Vec<Object<atrium_api::app::bsky::feed::defs::PostViewData>> =
            Vec::new();

        let thread_result = self
            .agent
            .api
            .app
            .bsky
            .feed
            .get_post_thread(
                get_post_thread::ParametersData {
                    uri: uri.to_string(),
                    depth: Some(LimitedU16::MAX), // We still need the full thread structure to trace parents
                    parent_height: None,
                }
                .into(),
            )
            .await?;

        match &thread_result.thread {
            // Match on a reference
            atrium_api::types::Union::Refs(r) => match r {
                get_post_thread::OutputThreadRefs::AppBskyFeedDefsThreadViewPost(object) => {
                    // Start collecting from the latest post (which is the root of this fetched thread)
                    self.collect_parents_recursive(object.clone(), &mut all_posts)?;
                }
                _ => return Err(anyhow::anyhow!("Unexpected thread type")),
            },
            _ => return Err(anyhow::anyhow!("Unexpected ref type")),
        };

        // The posts were collected from child to parent (latest to oldest),
        // so reverse to get chronological order (oldest to latest).
        all_posts.reverse();

        // Convert sorted posts to PostData
        let post_data: Vec<PostData> = all_posts
            .into_iter()
            .filter_map(|post| self.extract_post_data(post).ok())
            .collect();

        Ok(post_data)
    }

    /// Converts a collection of PostData objects to a JSON string
    ///
    /// This function serializes a vector of PostData objects into a JSON string,
    /// which can be used for storage, transmission, or display purposes.
    pub fn post_data_to_json_string(&self, post_data: &Vec<PostData>) -> Result<String> {
        serde_json::to_string(post_data)
            .map_err(|e| anyhow::anyhow!("Failed to serialize post data: {}", e))
    }

    /// Converts PostData objects to ChatMessage objects for LLM interaction
    ///
    /// This function transforms a vector of PostData objects into ChatMessage objects
    /// with the post content formatted as user messages for LLM processing.
    /// Convert post data to chat messages
    pub fn json_to_chatmessages(&self, post_data: Vec<PostData>) -> Vec<ChatMessage> {
        post_data
            .into_iter()
            .map(|post| {
                let mut message = format!("{}: {}", post.author, post.text);

                // Add embed information if present
                if let Some(embed) = post.embed {
                    match embed {
                        PostEmbed::Images(images) => {
                            message.push_str("\n[Images: ");
                            for (i, img) in images.images.iter().enumerate() {
                                if i > 0 {
                                    message.push_str(", ");
                                }
                                if let Some(alt) = &img.alt {
                                    message.push_str(&format!("\"{}\"", alt));
                                } else {
                                    message.push_str("image");
                                }
                            }
                            message.push_str("]");
                        }
                        PostEmbed::External(external) => {
                            message.push_str(&format!("\n[External link: {}]", external.uri));
                            if let Some(title) = external.title {
                                message.push_str(&format!(" - \"{}\"", title));
                            }
                        }
                        PostEmbed::Video(_) => {
                            message.push_str("\n[Video]");
                        }
                        PostEmbed::Record(record) => {
                            message.push_str(&format!("\n[Quoted post: {}]", record.record));
                        }
                        PostEmbed::RecordWithMedia(record_with_media) => {
                            message.push_str(&format!(
                                "\n[Quoted post with media: {}]",
                                record_with_media.record.record
                            ));
                        }
                    }
                }

                ChatMessage::user(message)
            })
            .collect()
    }

    /// Converts PostData objects to a single ChatMessage containing the JSON string
    ///
    /// Instead of creating individual ChatMessages for each post, this function
    /// serializes the entire collection of posts into a single JSON string and
    /// wraps it in a single ChatMessage. This is useful when you want to pass
    /// the structured data to an LLM that can parse JSON.
    pub fn json_to_stringified_chatmessages(
        &self,
        post_data: Vec<PostData>,
    ) -> Result<Vec<ChatMessage>> {
        let json_string = self.post_data_to_json_string(&post_data)?;
        Ok(vec![ChatMessage::user(json_string)])
    }

    /// Create a MemoryEntry from a PostData object
    ///
    /// This function converts a PostData object into a MemoryEntry that can be
    /// stored in the vector database. It creates a unique ID based on the post URI
    /// and uses the serialized PostData JSON as content for richer context.
    pub fn create_memory_entry_from_post(
        &self,
        post_data: &PostData,
        embedding: Vec<f32>,
    ) -> MemoryEntry {
        // Create a UUID based on the post URI
        let entry_id =
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, post_data.uri.as_bytes()).to_string();

        // Create a conversation ID from the author DID
        let conv_id =
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, post_data.author_did.as_bytes())
                .to_string();

        // Create a JSON string with the post data
        let content_json = serde_json::to_string(post_data)
            .unwrap_or_else(|_| format!("{}: {}", post_data.author, post_data.text));

        // Add additional tags based on embed content
        let mut tags = vec!["bluesky_post".to_string()];

        if let Some(embed) = &post_data.embed {
            match embed {
                PostEmbed::Images(_) => {
                    tags.push("has_images".to_string());
                }
                PostEmbed::External(_) => {
                    tags.push("has_external_link".to_string());
                }
                PostEmbed::Video(_) => {
                    tags.push("has_video".to_string());
                }
                PostEmbed::Record(_) => {
                    tags.push("has_quote".to_string());
                }
                PostEmbed::RecordWithMedia(_) => {
                    tags.push("has_quote_with_media".to_string());
                }
            }
        }

        MemoryEntry {
            id: entry_id,
            content: content_json,
            tags,
            embedding,
            conversation_id: conv_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            role: "user".to_string(),
            entry_type: "bluesky_post".to_string(),
        }
    }

    /// Fetches a thread and converts it directly to ChatMessages
    ///
    /// This is a convenience method that combines atp_thread_to_json() and
    /// json_to_chatmessages() to directly convert a thread to ChatMessage objects.
    /// This maintains backward compatibility with existing code.
    pub async fn atp_thread_to_chatmessages(&self, uri: &str) -> Result<Vec<ChatMessage>> {
        let post_data = self.atp_thread_to_json(uri).await?;
        Ok(self.json_to_chatmessages(post_data))
    }
}

/// A cool ingestor implementation.
#[async_trait]
impl LexiconIngestor for PostListener {
    async fn ingest(&self, message: Event<Value>) -> anyhow::Result<()> {
        // set up timer
        let timer = Instant::now();

        let result = async {
            if let Some(Commit {
                record: Some(record),
                cid: Some(cid),
                rkey,
                collection,
                ..
            }) = message.commit
            {
                let riposte = serde_json::from_value::<
                    atrium_api::app::bsky::feed::post::RecordData,
                >(record)?;

                let aturi = format!("at://{}/{}/{}", message.did, collection, rkey);

                let span = tracing::info_span!("PostListener::ingest", aturl = aturi);
                let _enter = span.enter();

                trace!("Processing post");

                trace!("recieved {}", riposte.text);

                // is user mentioning me or allowlisted
                if !self.is_me(riposte.clone()) || !self.is_allowlisted(&message.did) {
                    return Ok(());
                }

                trace!("replying...");

                // Extract thread as JSON data
                let post_data = self.atp_thread_to_json(&aturi).await?;

                // Convert to chat messages for LLM processing
                let thread = self.json_to_chatmessages(post_data.clone());

                trace!("{:?}", &thread);
                trace!("JSON post data: {:?}", &post_data);

                // Example of getting the stringified JSON version
                if let Ok(json_string) = self.post_data_to_json_string(&post_data) {
                    trace!("Stringified JSON: {}", json_string);
                }

                // Create memory entries from post data if we have any
                if !post_data.is_empty() {
                    // Log embeds for debugging
                    for post in &post_data {
                        if let Some(embed) = &post.embed {
                            trace!("Post has embed: {:?}", embed);
                        }
                    }

                    // Extract all post texts for embedding - include embed description when available
                    let post_texts: Vec<String> = post_data
                        .iter()
                        .map(|post| {
                            let mut text = post.text.clone();

                            // Add embed content to improve embeddings with this context
                            if let Some(embed) = &post.embed {
                                match embed {
                                    PostEmbed::External(external) => {
                                        if let Some(desc) = &external.description {
                                            text.push_str(" ");
                                            text.push_str(desc);
                                        }
                                        if let Some(title) = &external.title {
                                            text.push_str(" ");
                                            text.push_str(title);
                                        }
                                    }
                                    // Other embed types could be added here
                                    _ => {}
                                }
                            }

                            text
                        })
                        .collect();

                    // Generate embeddings for all posts in batch
                    if let Ok(embeddings) = self.emb.embed(post_texts) {
                        if !embeddings.is_empty() {
                            // Create memory entries for each post
                            let mut memory_entries = Vec::new();

                            for (i, post) in post_data.iter().enumerate() {
                                if i < embeddings.len() {
                                    let mem_entry = self
                                        .create_memory_entry_from_post(post, embeddings[i].clone());

                                    memory_entries.push(mem_entry);
                                }
                            }

                            trace!("Created {} memory entries", memory_entries.len());

                            // Here you would typically add these to your vector database
                            // Either individually:
                            // for entry in &memory_entries {
                            //     self.vdb.put(entry.clone()).await?;
                            // }
                            // Or as a batch if supported:
                            // self.vdb.put_batch(memory_entries).await?;
                        }
                    }
                }

                let texts: Vec<String> = thread
                    .iter()
                    .filter_map(|post| post.content.text_as_str().map(|s| s.to_string()))
                    .collect();

                let vecs = self.emb.embed(texts)?;

                // search db for similar posts
                let mut similar_posts = self
                    .vdb
                    .get_similar(
                        vecs.last().unwrap().to_owned(),
                        Some(vec!["stm".to_string()]),
                        2,
                    )
                    .await?;
                debug!("similar posts: {:?}", similar_posts);

                // Deduplicate by ID
                similar_posts.dedup_by_key(|p| p.id.clone());

                let mut search_chats_str = String::new();
                for entry in similar_posts {
                    search_chats_str.push_str(&format!("{}\n", entry.content));
                }

                debug!("search results: {:?}", &search_chats_str);

                let search_results_cm = ChatMessage::system(search_chats_str);

                // Create initial message array to send to the LLM
                let mut messages = thread.clone();
                messages.insert(0, search_results_cm);

                // Get initial response from LLM
                let initial_resp = self
                    .aisvc
                    .generate_response(&messages, None)
                    .await
                    .inspect(|x| println!("original: {x}"))?;

                // Process any tool calls in the response
                info!("Processing tool calls in LLM response...");

                // Tool call loop to allow chaining of multiple tool calls
                let mut response_accum = initial_resp.clone();
                let mut last_tool_call: Option<(String, serde_json::Value)> = None;
                let mut last_tool_call_times = 0;

                loop {
                    let tool_calls = parse_tool_calls(&response_accum);
                    if !tool_calls.is_empty() {
                        // Check for repeated tool call (name + args) for the first tool call only
                        let first_call = &tool_calls[0];
                        if let Some((last_name, last_args)) = &last_tool_call {
                            if last_name == &first_call.tool_name && last_args == &first_call.tool_args {
                                if last_tool_call_times >= TOOL_CALL_TIMES {
                                    debug!("Too many repeated tool calls, breaking to avoid infinite loop.");
                                    break;
                                }
                                last_tool_call_times += 1;
                            } else {
                                last_tool_call_times = 1; // Reset count for new tool call
                            }
                        } else {
                            last_tool_call_times = 1;
                        }
                        last_tool_call = Some((first_call.tool_name.clone(), first_call.tool_args.clone()));

                        // Execute all tool calls in order
                        debug!("Executing {} tool calls", tool_calls.len());
                        let tool_results = execute_tool_calls(&tool_calls, &self.tools).await;

                        // Add the assistant response to the conversation
                        messages.push(ChatMessage::assistant(response_accum.clone()));

                        // Add each tool response to the conversation
                        for (tool_name, result) in &tool_results {
                            match result {
                                Ok(tool_result) => {
                                    debug!("Tool '{}' returned: {}", tool_name, tool_result);
                                    messages.push(
                                        ToolResponse::new(tool_name.clone(), tool_result.to_string())
                                            .into(),
                                    );
                                }
                                Err(e) => {
                                    debug!("Tool '{}' error: {}", tool_name, e);
                                    messages.push(
                                        ToolResponse::new(tool_name.clone(), format!("Error: {}", e))
                                            .into(),
                                    );
                                }
                            }
                        }

                        // Get follow-up response
                        let followup_resp = self.aisvc.generate_response(&messages, None).await?;

                        // Prepare for next loop iteration
                        response_accum = followup_resp;
                        continue;
                    } else {
                        // No tool call, we're done
                        break;
                    }
                }

                let final_resp = response_accum;

                // remove <think> tag
                let resp = final_resp
                    .split("</think>")
                    .collect::<Vec<&str>>()
                    .last()
                    .ok_or(anyhow::anyhow!("no response outputted?"))?
                    .trim()
                    .to_string();

                // if response is empty, just return ok
                if resp.trim().is_empty() {
                    debug!("aigis doesn't want to reply, so not replying");
                    return Ok(());
                }

                // get the cid
                let rcid = match Cid::from_str(&cid) {
                    Ok(r) => r,
                    Err(e) => return Err(anyhow::anyhow!(e)),
                };

                let reply = self.build_reply_ref(
                    riposte.reply,
                    rcid,
                    message.did.clone(),
                    collection,
                    rkey,
                );

                // Get the URI from the reply for later use
                let root_uri = reply.root.uri.clone();

                self.agent
                    .create_record(atrium_api::app::bsky::feed::post::RecordData {
                        created_at: Datetime::now(),
                        embed: None,
                        entities: None,
                        facets: None,
                        labels: None,
                        langs: Some(vec![self.lang.clone()]),
                        reply: Some(reply),
                        tags: None,
                        text: resp.trim().to_string(),
                    })
                    .await?;

                let lm = thread.last().expect("thread has stuff in it");

                // put vector db stuff in struct
                let chat_log = ChatLog {
                    post: lm.content.clone().text_into_string().expect("text is some"),
                    response: resp.trim().to_string(),
                    poster_did: message.did.to_string(),
                };

                // serialize
                let chat_log = serde_json::to_string(&chat_log)?;

                // embed question + response
                let vector = self.emb.embed(vec![chat_log.clone()])?;

                // zip up vector and chat log

                let zipped = vector
                    .into_iter()
                    .zip(std::iter::once(chat_log.clone()))
                    .collect::<Vec<_>>();

                let mut memtries = Vec::new();

                for (vec, cl) in zipped {
                    // todo: use TID as the hash
                    let chatid =
                        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, cl.as_bytes()).to_string();

                    let convid =
                        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, root_uri.as_bytes())
                            .to_string();

                    let mem_entry = MemoryEntry {
                        id: chatid,
                        content: cl,
                        tags: vec!["stm".to_string()],
                        embedding: vec,
                        conversation_id: convid,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64,
                        role: "user".to_string(),
                        entry_type: "bluesky_post".to_string(),
                    };
                    memtries.push(mem_entry);
                }
            };
            Ok(())
        }
        .await;

        INGEST_LATENCY.record(timer.elapsed());

        match result {
            Ok(_) => {
                POSTS_INGESTED.increment(1);
                Ok(())
            }
            Err(e) => {
                INGEST_ERRORS.increment(1);
                error!(error = %e, "Failed to ingest post");
                Err(e)
            }
        }
    }
}

/// Represents basic post data for JSON serialization
///
/// This structure contains the essential information about a BlueSky post
/// in a format that can be easily serialized to JSON. It excludes complex
/// nested structures found in the original ATP/BlueSky data model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostData {
    /// Author name and handle (formatted as "Name (handle)")
    pub author: String,
    /// Content of the post (the actual text)
    pub text: String,
    /// Unique identifier for the post (AT URI)
    pub uri: String,
    /// Author's DID (Decentralized Identifier)
    pub author_did: String,
    /// Post timestamp as an ISO-8601 string
    pub indexed_at: Option<String>,

    pub embed: Option<PostEmbed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostEmbed {
    Images(PostEmbedImages),
    External(PostEmbedExternal),
    Video(PostEmbedVideo),
    Record(PostEmbedRecord),
    RecordWithMedia(PostEmbedRecordWithMedia),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEmbedImages {
    pub images: Vec<PostEmbedImage>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEmbedImage {
    pub image: String,       // URL to the image
    pub alt: Option<String>, // Optional alt text for the image
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEmbedExternal {
    pub uri: String,                 // URL to the external content
    pub title: Option<String>,       // Optional title for the external content
    pub description: Option<String>, // Optional description for the external content
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEmbedRecord {
    pub record: String,        // AT URI to the record
    pub title: Option<String>, // Optional title for the record
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEmbedRecordWithMedia {
    pub record: PostEmbedRecord,    // AT URI to the record
    pub media: Vec<PostEmbedMedia>, // Associated media images
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostEmbedMedia {
    Images(PostEmbedImages),
    Video(PostEmbedVideo),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEmbedVideo {
    pub video: String,         // URL to the video
    pub duration: Option<u64>, // Optional duration in seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatLog {
    post: String,
    response: String,
    poster_did: String,
}
