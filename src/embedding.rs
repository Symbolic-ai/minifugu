use crate::vector;
use serde_json::{json, Map, Value};

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

pub(crate) enum QueryEmbeddingError {
    Invalid(String),
    Unavailable,
}

struct QueryEmbedding {
    pointer: String,
    target: String,
    text: String,
    model: String,
    dims: usize,
}

/// Turn native Embed query operands into vectors before synchronous query validation.
/// Collect every request first so a malformed later clause never starts an embedding call.
pub(crate) async fn materialize_query(
    body: &mut Value,
    schema: &Map<String, Value>,
    mode: &EmbeddingMode,
) -> Result<(), QueryEmbeddingError> {
    let mut requests = Vec::new();
    collect_query_embeddings(body, "", schema, &mut requests)
        .map_err(QueryEmbeddingError::Invalid)?;
    for request in requests {
        let vector = mode
            .embed(&request.text, &request.model, request.dims)
            .await
            .map_err(|()| QueryEmbeddingError::Unavailable)?;
        let parts = body
            .pointer_mut(&request.pointer)
            .and_then(Value::as_array_mut)
            .expect("collected query expression remains present");
        parts[0] = json!(request.target);
        parts[2] = json!(vector);
    }
    Ok(())
}

fn collect_query_embeddings(
    value: &Value,
    pointer: &str,
    schema: &Map<String, Value>,
    requests: &mut Vec<QueryEmbedding>,
) -> Result<(), String> {
    if let Some(parts) = value.as_array() {
        if parts.len() == 3
            && matches!(parts[1].as_str(), Some("ANN" | "kNN"))
            && parts[2]
                .as_array()
                .is_some_and(|operand| operand.first() == Some(&json!("Embed")))
        {
            let field = parts[0]
                .as_str()
                .ok_or("Embed target must be an attribute")?;
            let operand = parts[2].as_array().unwrap();
            if !(2..=3).contains(&operand.len()) {
                return Err("Embed requires text and optional model parameters".into());
            }
            let text = operand[1].as_str().ok_or("Embed text must be a string")?;
            let explicit_model = if operand.len() == 3 {
                let options = operand[2]
                    .as_object()
                    .ok_or("Embed parameters must be an object")?;
                if options.keys().any(|key| key != "model") {
                    return Err("unsupported Embed parameter".into());
                }
                options
                    .get("model")
                    .map(|model| model.as_str().ok_or("Embed model must be a string"))
                    .transpose()?
            } else {
                None
            };
            let source = schema
                .get(field)
                .and_then(|definition| definition.get("embed"));
            let (target, dims) = if let Some(config) = source {
                (
                    format!("embed_{field}"),
                    config.get("dims").and_then(Value::as_u64).unwrap_or(1536) as usize,
                )
            } else if let Some(base) = field.strip_prefix("embed_") {
                let config = schema
                    .get(base)
                    .and_then(|definition| definition.get("embed"))
                    .ok_or_else(|| format!("attribute {field} is not a vector"))?;
                (
                    field.to_owned(),
                    config.get("dims").and_then(Value::as_u64).unwrap_or(1536) as usize,
                )
            } else {
                let dims = schema
                    .get(field)
                    .and_then(vector::dimensions)
                    .ok_or_else(|| format!("attribute {field} is not a vector"))?;
                (field.to_owned(), dims)
            };
            let model = explicit_model
                .or_else(|| source.and_then(|config| config.get("model").and_then(Value::as_str)))
                .ok_or(
                    "a model name must be provided when ranking a vector by an embedding query",
                )?;
            requests.push(QueryEmbedding {
                pointer: pointer.to_owned(),
                target,
                text: text.to_owned(),
                model: model.to_owned(),
                dims,
            });
            return Ok(());
        }
        for (index, child) in parts.iter().enumerate() {
            collect_query_embeddings(child, &format!("{pointer}/{index}"), schema, requests)?;
        }
    } else if let Some(object) = value.as_object() {
        for (key, child) in object {
            let key = key.replace('~', "~0").replace('/', "~1");
            collect_query_embeddings(child, &format!("{pointer}/{key}"), schema, requests)?;
        }
    }
    Ok(())
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
