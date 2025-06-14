use qdrant_client::{
    Qdrant,
    config::QdrantConfig,
    qdrant::{
        CreateCollectionBuilder, Distance, HnswConfigDiffBuilder, PointStruct, ScoredPoint, SearchBatchPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
        Value, VectorParamsBuilder,
    },
};
use std::collections::HashMap;
use tracing::debug;

pub struct VectorDb {
    client: Qdrant,
    collection_name: String,
}

impl VectorDb {
    pub async fn new(url: &str, collection_name: &str) -> anyhow::Result<Self> {
        let config = QdrantConfig::from_url(url);
        let client = Qdrant::new(config)?;

        // if collection doesn't exist, create it
        if let Err(_) = client.collection_info(collection_name).await {
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection_name)
                        .on_disk_payload(true)
                        .hnsw_config(HnswConfigDiffBuilder::default().on_disk(true))
                        .vectors_config(VectorParamsBuilder::new(384, Distance::Cosine)),
                )
                .await?;
        }

        Ok(VectorDb {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    pub async fn store_memory(
        &self,
        id: &str,
        vector: Vec<f32>,
        payload: &str,
    ) -> anyhow::Result<()> {
        let payload_map: HashMap<String, Value> =
            HashMap::from([("text".to_string(), Value::from(payload))]);

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.collection_name,
                vec![PointStruct::new(id, vector, payload_map)],
            ))
            .await?;

        Ok(())
    }

    pub async fn search_similar(
        &self,
        query_vector: Vec<f32>,
        top: usize,
    ) -> anyhow::Result<Vec<ScoredPoint>> {
        let search_result = self
            .client
            .search_points(SearchPointsBuilder::new(
                &self.collection_name,
                query_vector,
                top as u64,
            ))
            .await?;

        debug!("Search result: {:?}", &search_result.result);

        Ok(search_result.result)
    }

    pub async fn batch_search_similar(
        &self,
        query_vectors: Vec<Vec<f32>>,
        top: usize,
    ) -> anyhow::Result<Vec<ScoredPoint>> {
        let sbp_vecs = query_vectors
            .into_iter()
            .map(|vec| SearchPointsBuilder::new(&self.collection_name, vec, top as u64).build())
            .collect::<Vec<_>>();

        let batch = SearchBatchPointsBuilder::new(&self.collection_name, sbp_vecs);

        let res = self.client.search_batch_points(batch).await?;

        let unique_results = res.result.into_iter().flat_map(|r| r.result).collect();

        Ok(unique_results)
    }
}
