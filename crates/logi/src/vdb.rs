use qdrant_client::{
    config::QdrantConfig,
    qdrant::{
        r#match::MatchValue, CreateCollectionBuilder, Distance, FieldCondition, Filter,
        HnswConfigDiffBuilder, Match, PointStruct, ScoredPoint, SearchPointsBuilder,
        UpsertPointsBuilder, Value, VectorParamsBuilder,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub timestamp: i64, // unix timestamp in seconds
    pub tags: Vec<String>,
    pub role: String,       // "user" or "assistant"
    pub entry_type: String, // e.g. "message", "fact", "tool_result"
    pub conversation_id: String,
}

pub struct MemoryStore {
    client: Qdrant,
    collection_name: String,
}

impl MemoryStore {
    pub async fn new(
        url: &str,
        collection_name: &str,
        embedding_dim: usize,
    ) -> anyhow::Result<Self> {
        let config = QdrantConfig::from_url(url);
        let client = Qdrant::new(config)?;

        // Create collection if it doesn't exist
        if let Err(_) = client.collection_info(collection_name).await {
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection_name)
                        .on_disk_payload(true)
                        .hnsw_config(HnswConfigDiffBuilder::default().on_disk(true))
                        .vectors_config(VectorParamsBuilder::new(
                            embedding_dim as u64,
                            Distance::Cosine,
                        )),
                )
                .await?;
        }

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    pub async fn put(&self, entry: MemoryEntry) -> anyhow::Result<()> {
        let mut payload_map: HashMap<String, Value> = HashMap::new();
        payload_map.insert("content".to_string(), Value::from(entry.content.clone()));
        payload_map.insert("timestamp".to_string(), Value::from(entry.timestamp));
        payload_map.insert("tags".to_string(), Value::from(entry.tags.clone()));
        payload_map.insert("role".to_string(), Value::from(entry.role.clone()));
        payload_map.insert(
            "entry_type".to_string(),
            Value::from(entry.entry_type.clone()),
        );
        payload_map.insert(
            "conversation_id".to_string(),
            Value::from(entry.conversation_id.clone()),
        );

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.collection_name,
                vec![PointStruct::new(entry.id, entry.embedding, payload_map)],
            ))
            .await?;

        Ok(())
    }

    pub async fn put_batch(&self, entries: Vec<MemoryEntry>) -> anyhow::Result<()> {
        let points: Vec<PointStruct> = entries
            .into_iter()
            .map(|entry| {
                let mut payload_map: HashMap<String, Value> = HashMap::new();
                payload_map.insert("content".to_string(), Value::from(entry.content.clone()));
                payload_map.insert("timestamp".to_string(), Value::from(entry.timestamp));
                payload_map.insert("tags".to_string(), Value::from(entry.tags.clone()));
                payload_map.insert("role".to_string(), Value::from(entry.role.clone()));
                payload_map.insert(
                    "entry_type".to_string(),
                    Value::from(entry.entry_type.clone()),
                );
                payload_map.insert(
                    "conversation_id".to_string(),
                    Value::from(entry.conversation_id.clone()),
                );
                PointStruct::new(entry.id, entry.embedding, payload_map)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await?;

        Ok(())
    }

    pub async fn get_similar(
        &self,
        embedding: Vec<f32>,
        tags: Option<Vec<String>>,
        top_k: usize,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        // Build filter for tags if provided
        let filter = tags.map(|tags| Filter {
            must: tags
                .into_iter()
                .map(|tag| {
                    FieldCondition {
                        key: "tags".to_string(),
                        r#match: Some(MatchValue::Keyword(tag).into()),
                        ..Default::default()
                    }
                    .into()
                })
                .collect(),
            ..Default::default()
        });

        let mut builder = SearchPointsBuilder::new(&self.collection_name, embedding, top_k as u64)
            .with_payload(true);

        if let Some(f) = filter {
            builder = builder.filter(f);
        }

        let search_result = self.client.search_points(builder).await?;
        debug!("Search result: {:?}", &search_result.result);

        // Deserialize results into MemoryEntry
        let mut entries = Vec::new();
        for point in search_result.result {
            let entry: MemoryEntry = serde_json::from_value(serde_json::to_value(point.payload)?)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    pub async fn get_by_filter(&self, filter: Filter) -> anyhow::Result<Vec<MemoryEntry>> {
        // Use a dummy vector and large top_k to fetch by filter only
        let builder = SearchPointsBuilder::new(&self.collection_name, vec![0.0; 1], 100)
            .with_payload(true)
            .filter(filter);

        let search_result = self.client.search_points(builder).await?;
        let mut entries = Vec::new();
        for point in search_result.result {
            let entry: MemoryEntry = serde_json::from_value(serde_json::to_value(point.payload)?)?;
            entries.push(entry);
        }
        Ok(entries)
    }

    pub async fn get_pair(
        &self,
        query_embedding: Vec<f32>,
    ) -> anyhow::Result<Option<(MemoryEntry, Option<MemoryEntry>)>> {
        // Step 1: Find most similar user message
        let user_msgs = self
            .get_similar(query_embedding, Some(vec!["user".to_string()]), 1)
            .await?;
        let user_msg = match user_msgs.into_iter().next() {
            Some(m) => m,
            None => return Ok(None),
        };

        // Step 2: Find the next assistant message in the same conversation
        let filter = Filter {
            must: vec![
                FieldCondition {
                    key: "conversation_id".to_string(),
                    r#match: Some(MatchValue::Keyword(user_msg.conversation_id.clone()).into()),
                    ..Default::default()
                }
                .into(),
                FieldCondition {
                    key: "role".to_string(),
                    r#match: Some(MatchValue::Keyword("assistant".to_string()).into()),
                    ..Default::default()
                }
                .into(),
            ],
            ..Default::default()
        };
        let mut convo_points = self.get_by_filter(filter).await?;
        convo_points.sort_by_key(|m| m.timestamp);

        // Find the first assistant message after the user message
        let assistant_msg = convo_points
            .into_iter()
            .find(|m| m.timestamp > user_msg.timestamp);

        Ok(Some((user_msg, assistant_msg)))
    }

    pub async fn get_chain(&self, conversation_id: &str) -> anyhow::Result<Vec<MemoryEntry>> {
        let filter = Filter {
            must: vec![FieldCondition {
                key: "conversation_id".to_string(),
                r#match: Some(MatchValue::Keyword(conversation_id.to_string()).into()),
                ..Default::default()
            }
            .into()],
            ..Default::default()
        };
        let mut convo_points = self.get_by_filter(filter).await?;
        convo_points.sort_by_key(|m| m.timestamp);
        Ok(convo_points)
    }
}
