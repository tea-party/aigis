use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct Embedder {
    embedder: TextEmbedding,
}

impl Embedder {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            embedder: TextEmbedding::try_new(InitOptions::new(
                EmbeddingModel::ParaphraseMLMiniLML12V2,
            ))?,
        })
    }
    pub fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let embeddings = self
            .embedder
            .embed(texts, None)
            .map_err(|e| anyhow::anyhow!("Failed to embed: {}", e))?;
        Ok(embeddings)
    }
}
