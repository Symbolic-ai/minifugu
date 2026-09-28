#![forbid(unsafe_code)]

mod embedding;
mod persistence;
mod query;
mod store;

use axum::{
    extract::{Path, State},
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::{Path as FilePath, PathBuf},
    sync::Arc,
};
use tokio::sync::RwLock;

pub use embedding::{deterministic_embedding, EmbeddingMode};
pub use store::Namespace;
type Shared = Arc<AppState>;
struct AppState {
    namespaces: RwLock<HashMap<String, Namespace>>,
    embedding: EmbeddingMode,
    data_path: Option<PathBuf>,
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
    router_with_state(embedding, None, HashMap::new())
}

pub fn router_with_data_dir(
    embedding: EmbeddingMode,
    directory: &FilePath,
) -> std::io::Result<Router> {
    let (path, namespaces) = persistence::open(directory)?;
    Ok(router_with_state(embedding, Some(path), namespaces))
}

fn router_with_state(
    embedding: EmbeddingMode,
    data_path: Option<PathBuf>,
    namespaces: HashMap<String, Namespace>,
) -> Router {
    let state = Arc::new(AppState {
        namespaces: RwLock::new(namespaces),
        embedding,
        data_path,
    });
    Router::new()
        .route("/v1/namespaces", get(list_namespaces))
        .route(
            "/v1/namespaces/{namespace}/schema",
            get(get_schema).post(update_schema),
        )
        .route("/v2/namespaces/{namespace}/metadata", get(get_metadata))
        .route(
            "/v2/namespaces/{namespace}",
            post(write).delete(delete_namespace),
        )
        .route("/v2/namespaces/{namespace}/query", post(query_namespace))
        .with_state(state)
}

async fn list_namespaces(
    State(state): State<Shared>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    let guard = state.namespaces.read().await;
    let mut names = guard.keys().collect::<Vec<_>>();
    names.sort();
    Ok(Json(
        json!({"namespaces": names.iter().map(|id| json!({"id":id})).collect::<Vec<_>>()}),
    ))
}

async fn get_schema(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let guard = state.namespaces.read().await;
    let namespace = guard
        .get(&name)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    Ok(Json(json!(namespace.schema)))
}

async fn get_metadata(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let guard = state.namespaces.read().await;
    let namespace = guard
        .get(&name)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    Ok(Json(
        json!({"schema":namespace.schema,"approx_row_count":namespace.rows.len(),"index":{"status":"up-to-date"},"read_only":false}),
    ))
}

async fn update_schema(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    Json(schema): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let mut guard = state.namespaces.write().await;
    let mut namespace = guard
        .get(&name)
        .cloned()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    namespace
        .write(&json!({"schema":schema}), &state.embedding)
        .await
        .map_err(|error| match error {
            store::WriteError::Invalid(message) => bad(message),
            store::WriteError::EmbeddingUnavailable => bad("embedding provider unavailable"),
        })?;
    let response = json!(namespace.schema);
    persist_namespace(&state, &mut guard, name, namespace)?;
    Ok(Json(response))
}

fn persist_namespace(
    state: &AppState,
    guard: &mut HashMap<String, Namespace>,
    name: String,
    namespace: Namespace,
) -> Result<(), ApiError> {
    if let Some(path) = &state.data_path {
        let mut next = guard.clone();
        next.insert(name, namespace);
        persistence::save(path, &next).map_err(|_| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to persist namespace".into(),
            )
        })?;
        *guard = next;
    } else {
        guard.insert(name, namespace);
    }
    Ok(())
}

async fn write(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let object = body
        .as_object()
        .ok_or_else(|| bad("write body must be an object"))?;
    store::validate_write_keys(object).map_err(bad)?;
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
    persist_namespace(&state, &mut guard, name, namespace)?;
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
    if !guard.contains_key(&name) {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "namespace does not exist".into(),
        ));
    }
    if let Some(path) = &state.data_path {
        let mut next = guard.clone();
        next.remove(&name);
        persistence::save(path, &next).map_err(|_| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to persist namespace".into(),
            )
        })?;
        *guard = next;
    } else {
        guard.remove(&name);
    }
    Ok(Json(json!({"status":"ok"})))
}
