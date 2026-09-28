mod embedding;
mod query;
mod store;

use axum::{
    extract::{Path, State},
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

pub use embedding::{deterministic_embedding, EmbeddingMode};
pub use store::Namespace;
type Shared = Arc<AppState>;
struct AppState {
    namespaces: RwLock<HashMap<String, Namespace>>,
    embedding: EmbeddingMode,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(json!({"status":"error", "error":self.1}))).into_response()
    }
}

fn bad(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}

fn authorized(headers: &HeaderMap) -> Result<(), ApiError> {
    let value = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok());
    if value.is_some_and(|v| v.starts_with("Bearer ") && !v[7..].trim().is_empty()) {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "missing bearer token".into(),
        ))
    }
}

fn namespace_name(name: &str) -> Result<(), ApiError> {
    if name.len() > 128
        || name.is_empty()
        || matches!(name, "." | "..")
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
    {
        Err(bad("invalid namespace name"))
    } else {
        Ok(())
    }
}

pub fn router() -> Router {
    router_with_mode(EmbeddingMode::Deterministic)
}

pub fn router_with_mode(embedding: EmbeddingMode) -> Router {
    let state = Arc::new(AppState {
        namespaces: RwLock::new(HashMap::new()),
        embedding,
    });
    Router::new()
        .route(
            "/v2/namespaces/{namespace}",
            post(write).delete(delete_namespace),
        )
        .route("/v2/namespaces/{namespace}/query", post(query_namespace))
        .with_state(state)
}

async fn write(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let mut guard = state.namespaces.write().await;
    if !guard.contains_key(&name)
        && body
            .get("upsert_rows")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
    {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "namespace does not exist".into(),
        ));
    }
    let mut namespace = guard.get(&name).cloned().unwrap_or_default();
    let result = namespace
        .write(&body, &state.embedding)
        .await
        .map_err(|error| match error {
            store::WriteError::Invalid(message) => bad(message),
            store::WriteError::EmbeddingUnavailable => ApiError(
                StatusCode::BAD_GATEWAY,
                "embedding provider unavailable".into(),
            ),
        })?;
    guard.insert(name, namespace);
    Ok(Json(result))
}

async fn query_namespace(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let guard = state.namespaces.read().await;
    let namespace = guard.get(&name).ok_or_else(|| {
        ApiError(
            StatusCode::NOT_FOUND,
            format!("namespace {name} does not exist"),
        )
    })?;
    namespace.query(&body).map(Json).map_err(bad)
}

async fn delete_namespace(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let mut guard = state.namespaces.write().await;
    if guard.remove(&name).is_none() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "namespace does not exist".into(),
        ));
    }
    Ok(Json(json!({"status":"ok"})))
}
