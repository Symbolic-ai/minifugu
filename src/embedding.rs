use crate::store::embedding_target;
use crate::vector;
use serde_json::{json, Map, Value};

#[derive(Clone)]
pub enum EmbeddingMode {
    /// Stable token-hash vectors for offline tests. They are not semantic embeddings.
    Deterministic,
    /// Calls the OpenAI embeddings API for native `embed` schema fields.
    OpenAI { api_key: String, base_url: String },
}

pub(crate) enum EmbeddingError {
    InvalidModel,
    Unavailable,
}

impl EmbeddingMode {
    pub(crate) async fn embed(
        &self,
        text: &str,
        model: &str,
        dims: usize,
    ) -> Result<Vec<f32>, EmbeddingError> {
        match self {
            Self::Deterministic => Ok(deterministic_embedding(text, dims)),
            Self::OpenAI { api_key, base_url } => {
                let model = model
                    .strip_prefix("openai/")
                    .ok_or(EmbeddingError::InvalidModel)?;
                let response = reqwest::Client::new()
                    .post(format!("{}/v1/embeddings", base_url.trim_end_matches('/')))
                    .bearer_auth(api_key)
                    .json(&json!({"model":model,"input":text,"dimensions":dims,"encoding_format":"float"}))
                    .send().await.map_err(|_| EmbeddingError::Unavailable)?;
                if !response.status().is_success() {
                    return Err(
                        if response.status().is_client_error()
                            && !matches!(response.status().as_u16(), 401 | 403)
                        {
                            EmbeddingError::InvalidModel
                        } else {
                            EmbeddingError::Unavailable
                        },
                    );
                }
                let body: Value = response
                    .json()
                    .await
                    .map_err(|_| EmbeddingError::Unavailable)?;
                let values = body
                    .pointer("/data/0/embedding")
                    .and_then(Value::as_array)
                    .ok_or(EmbeddingError::Unavailable)?;
                let result = values
                    .iter()
                    .map(|v| v.as_f64().map(|n| n as f32))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(EmbeddingError::Unavailable)?;
                if result.len() != dims {
                    return Err(EmbeddingError::InvalidModel);
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

pub(crate) struct QueryEmbedding {
    pointer: String,
    target: String,
    text: String,
    model: String,
    dims: usize,
}

/// Replace query-time Embed operands with vectors of the right shape so the ordinary
/// query validator can reject malformed requests before any provider call starts.
pub(crate) fn prepare_query(
    body: &mut Value,
    schema: &Map<String, Value>,
) -> Result<Vec<QueryEmbedding>, String> {
    let object = body.as_object().ok_or("query body must be an object")?;
    let mut requests = Vec::new();
    if let Some(queries) = object.get("queries") {
        let queries = queries
            .as_array()
            .ok_or_else(|| crate::shape_error("queries must be an array"))?;
        if queries.is_empty() || queries.len() > 16 {
            return Err("queries must contain 1 to 16 subqueries".into());
        }
        for (index, query) in queries.iter().enumerate() {
            let query = query.as_object().ok_or("subquery must be an object")?;
            collect_query_object(query, &format!("/queries/{index}"), schema, &mut requests)?;
        }
    } else {
        collect_query_object(object, "", schema, &mut requests)?;
    }
    if requests.len() > 16 {
        return Err("query contains more than 16 Embed clauses".into());
    }
    for request in &requests {
        let parts = body
            .pointer_mut(&request.pointer)
            .and_then(Value::as_array_mut)
            .expect("collected query expression remains present");
        parts[0] = json!(request.target);
        parts[2] = json!(vec![0.0; request.dims]);
    }
    Ok(requests)
}

fn collect_query_object(
    object: &Map<String, Value>,
    pointer: &str,
    schema: &Map<String, Value>,
    requests: &mut Vec<QueryEmbedding>,
) -> Result<(), String> {
    if let Some(rank) = object.get("rank_by") {
        collect_query_embeddings(rank, &format!("{pointer}/rank_by"), schema, requests)?;
    }
    if let Some(computed) = object.get("compute_attributes").and_then(Value::as_object) {
        for (name, expression) in computed {
            let name = name.replace('~', "~0").replace('/', "~1");
            collect_query_embeddings(
                expression,
                &format!("{pointer}/compute_attributes/{name}"),
                schema,
                requests,
            )?;
        }
    }
    Ok(())
}

/// Embed a validated request. The clause cap also bounds provider concurrency.
pub(crate) async fn materialize_query(
    body: &mut Value,
    requests: Vec<QueryEmbedding>,
    mode: &EmbeddingMode,
) -> Result<(), QueryEmbeddingError> {
    let mut pending = tokio::task::JoinSet::new();
    for (index, request) in requests.iter().enumerate() {
        let mode = mode.clone();
        let text = request.text.clone();
        let model = request.model.clone();
        let dims = request.dims;
        pending.spawn(async move { (index, mode.embed(&text, &model, dims).await) });
    }
    let mut vectors = vec![None; requests.len()];
    while let Some(outcome) = pending.join_next().await {
        let (index, result) = outcome.map_err(|_| QueryEmbeddingError::Unavailable)?;
        let vector = result.map_err(|error| match error {
            EmbeddingError::InvalidModel => QueryEmbeddingError::Invalid(format!(
                "invalid embedding model or dimensions: {}",
                requests[index].model
            )),
            EmbeddingError::Unavailable => QueryEmbeddingError::Unavailable,
        })?;
        vectors[index] = Some(vector);
    }
    for (request, vector) in requests.into_iter().zip(vectors) {
        let parts = body
            .pointer_mut(&request.pointer)
            .and_then(Value::as_array_mut)
            .expect("prepared query expression remains present");
        parts[2] = json!(vector.expect("each embedding request completed"));
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
            && matches!(parts[1].as_str(), Some("ANN" | "kNN" | "VectorDist"))
            && parts[2]
                .as_array()
                .is_some_and(|operand| operand.first() == Some(&json!("Embed")))
        {
            if parts[1] == "VectorDist" {
                return Err(crate::shape_error(
                    "Embed is not supported in compute_attributes VectorDist",
                ));
            }
            let field = parts[0]
                .as_str()
                .ok_or_else(|| crate::shape_error("Embed target must be an attribute"))?;
            let operand = parts[2].as_array().unwrap();
            if !(2..=3).contains(&operand.len()) {
                return Err(crate::shape_error(
                    "Embed requires text and optional model parameters",
                ));
            }
            let text = operand[1]
                .as_str()
                .ok_or_else(|| crate::shape_error("Embed text must be a string"))?;
            let explicit_model = if operand.len() == 3 {
                let options = operand[2]
                    .as_object()
                    .ok_or_else(|| crate::shape_error("Embed parameters must be an object"))?;
                options
                    .get("model")
                    .filter(|model| !model.is_null())
                    .map(|model| {
                        model
                            .as_str()
                            .ok_or_else(|| crate::shape_error("Embed model must be a string"))
                    })
                    .transpose()?
            } else {
                None
            };
            let source = schema
                .get(field)
                .and_then(|definition| definition.get("embed"));
            let (target, dims) = if let Some(config) = source {
                (
                    embedding_target(field, config),
                    config.get("dims").and_then(Value::as_u64).unwrap_or(1536) as usize,
                )
            } else if let Some(config) = field
                .strip_prefix("embed_")
                .and_then(|base| Some((base, schema.get(base)?.get("embed")?)))
                // A source with a named target does not write embed_<source>.
                .filter(|(base, config)| embedding_target(base, config) == field)
                .map(|(_, config)| config)
            {
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
