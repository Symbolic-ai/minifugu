use serde_json::{json, Value};

#[derive(Clone)]
pub enum EmbeddingMode {
    /// Stable token-hash vectors for offline tests. They are not semantic embeddings.
    Deterministic,
    /// Calls the OpenAI embeddings API for native `embed` schema fields.
    OpenAI { api_key: String, base_url: String },
}

impl EmbeddingMode {
    pub(crate) async fn embed(&self, text: &str, model: &str, dims: usize) -> Result<Vec<f32>, ()> {
        match self {
            Self::Deterministic => Ok(deterministic_embedding(text, dims)),
            Self::OpenAI { api_key, base_url } => {
                let model = model.strip_prefix("openai/").ok_or(())?;
                let response = reqwest::Client::new()
                    .post(format!("{}/v1/embeddings", base_url.trim_end_matches('/')))
                    .bearer_auth(api_key)
                    .json(&json!({"model":model,"input":text,"dimensions":dims,"encoding_format":"float"}))
                    .send().await.map_err(|_| ())?;
                if !response.status().is_success() {
                    return Err(());
                }
                let body: Value = response.json().await.map_err(|_| ())?;
                let values = body
                    .pointer("/data/0/embedding")
                    .and_then(Value::as_array)
                    .ok_or(())?;
                let result = values
                    .iter()
                    .map(|v| v.as_f64().map(|n| n as f32))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(())?;
                if result.len() != dims {
                    return Err(());
                }
                Ok(result)
            }
        }
    }
}

// FNV-1a token hashing keeps offline vectors stable without network calls.
pub fn deterministic_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dims];
    if dims == 0 {
        return vector;
    }
    for token in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let hash = token
            .to_lowercase()
            .bytes()
            .fold(0xcbf29ce484222325_u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
        vector[(hash % dims as u64) as usize] += if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
    }
    let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for item in &mut vector {
            *item /= magnitude;
        }
    }
    vector
}
