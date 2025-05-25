use std::{
    collections::HashMap,
    str::FromStr,
    sync::{Arc, Mutex},
};

use ai::{AiService, LLMService};
use anyhow::Result;
use atrium_api::{
    app::bsky::feed::{
        defs::{PostViewData, ThreadViewPostData},
        get_post_thread,
        post::ReplyRefData,
    },
    com::atproto::repo::strong_ref::MainData,
    types::{
        LimitedU16, Object, TryFromUnknown,
        string::{Cid, Datetime, Did, Language},
    },
};
use bsky_sdk::BskyAgent;
use cursor::load_cursor;
use genai::chat::ChatMessage;
use metrics_exporter_prometheus::PrometheusBuilder;
use serde_json::Value;
use tracing::{debug, error, info};

use rocketman::{
    connection::JetstreamConnection,
    handler,
    ingestion::LexiconIngestor,
    options::JetstreamOptions,
    types::event::{Commit, Event},
};

use async_trait::async_trait;

mod ai;
mod cursor;
mod embed;
mod ingestors;
mod vdb;

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
}

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
    setup_tracing();
    setup_metrics();
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
            if users.is_empty() { None } else { Some(users) }
        }
        _ => None,
    };

    let system_message = match std::fs::read_to_string("./prompt.txt") {
        Ok(content) => Some(content),
        Err(e) => {
            error!("Could not read ./prompt.txt: {}", e);
            info!("Using default system message for AI service.");
            None
        }
    };

    ingestors.insert(
        // your EXACT nsid
        "app.bsky.feed.post".to_string(),
        Box::new(PostListener::new(
            agent.clone(),
            did,
            allowlist,
            system_message,
        )),
    );

    // tracks the last message we've processed
    let cursor: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(load_cursor().await));

    // get channels
    let msg_rx = jetstream.get_msg_rx();
    let reconnect_tx = jetstream.get_reconnect_tx();

    // spawn a task to process messages from the queue.
    // this is a simple implementation, you can use a more complex one based on needs.
    let c_cursor = cursor.clone();
    tokio::spawn(async move {
        while let Ok(message) = msg_rx.recv_async().await {
            if let Err(e) =
                handler::handle_message(message, &ingestors, reconnect_tx.clone(), c_cursor.clone())
                    .await
            {
                eprintln!("Error processing message: {}", e);
            };
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
    }
}

pub struct PostListener {
    agent: BskyAgent,
    did: Did,
    did_string: String,
    lang: Language,
    allowlist: Option<Vec<String>>,
    aisvc: LLMService, // llm svc
                       // emb: Embedder,
                       // vdb: VectorDb,
}

impl PostListener {
    pub fn new(
        agent: BskyAgent,
        did: Did,
        allowlist: Option<Vec<String>>,
        system_message: Option<String>,
    ) -> Self {
        let lang = Language::from_str("en").unwrap();
        let default_system_message = "You are an unhelpful assistant, named Aigis, that acts dumb and stupid. type in lowercase, and never ever add *actions* or use emojis as they are uncool. Don't mention Persona 3 topics unless explicitly asked about it. be extremely dumb and stupid. The messages you'll see are chat logs from a platform called Bluesky, they are in the format `username (handle): message`, but you only should reply to them with the message. Keep under a strict limit of 250 characters.";
        let aisvc = LLMService::new(system_message.as_deref().or(Some(default_system_message)))
            .expect("LLM Service initiated");
        Self {
            agent,
            did_string: did.clone().to_string(),
            did,
            lang,
            allowlist,
            aisvc,
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

        return Ok(format!("{}: {}", author, record_data.text));
    }

    // New helper function to collect posts by traversing up the parent chain
    fn collect_parents_recursive(
        &self,
        current_thread_view: Box<Object<ThreadViewPostData>>,
        posts: &mut Vec<Object<PostViewData>>,
    ) -> Result<()> {
        // Add the current post to the list
        posts.push((*current_thread_view).post.clone());

        // Check if there's a parent and if it's a ThreadViewPost
        if let Some(parent_thread_view) = (*current_thread_view).parent.clone() {
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

    pub async fn atp_thread_to_chatmessages(&self, uri: &str) -> Result<Vec<ChatMessage>> {
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

        // Convert sorted posts to ChatMessage
        let messages: Vec<ChatMessage> = all_posts
            .into_iter()
            .filter_map(|post| self.assemble_post_message(post).ok())
            .map(ChatMessage::user)
            .collect();

        Ok(messages)
    }
}

/// A cool ingestor implementation.
#[async_trait]
impl LexiconIngestor for PostListener {
    async fn ingest(&self, message: Event<Value>) -> anyhow::Result<()> {
        if let Some(Commit {
            record: Some(record),
            cid: Some(cid),
            rkey,
            collection,
            ..
        }) = message.commit
        {
            let riposte =
                serde_json::from_value::<atrium_api::app::bsky::feed::post::RecordData>(record)?;

            debug!("recieved {}", riposte.text);

            // is user mentioning me or allowlisted
            if !self.is_me(riposte.clone()) || !self.is_allowlisted(&message.did) {
                return Ok(());
            }

            debug!("replying...");

            // get + build post thread by tracing parents
            let thread = self
                .atp_thread_to_chatmessages(&format!(
                    "at://{}/{}/{}",
                    message.did, collection, rkey
                ))
                .await?;

            dbg!(&thread);

            let initial_resp = self
                .aisvc
                .generate_response(thread)
                .await
                .inspect(|x| println!("original: {x}"))?;

            // remove <think> tag
            let resp = initial_resp
                .split("</think>")
                .collect::<Vec<&str>>()
                .last()
                .ok_or(anyhow::anyhow!("no response outputted?"))?
                .to_string();

            // get the cid
            let rcid = match Cid::from_str(&cid) {
                Ok(r) => r,
                Err(e) => return Err(anyhow::anyhow!(e)),
            };

            let reply = self.build_reply_ref(riposte.reply, rcid, message.did, collection, rkey);

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
        }
        Ok(())
    }
}
