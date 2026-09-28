#![forbid(unsafe_code)]

mod embedding;
mod persistence;
mod query;
mod store;

use axum::{
    extract::{Path, Query, State},
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
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    let page_size = params.get("page_size").map_or(Ok(1000), |raw| {
        raw.parse::<usize>().map_err(|_| bad("invalid page_size"))
    })?;
    if !(1..=1000).contains(&page_size) {
        return Err(bad("page_size must be between 1 and 1000"));
    }
    let guard = state.namespaces.read().await;
    let mut names = guard
        .keys()
        .filter(|name| {
            params
                .get("prefix")
                .is_none_or(|prefix| name.starts_with(prefix))
        })
        .filter(|name| params.get("cursor").is_none_or(|cursor| *name > cursor))
        .collect::<Vec<_>>();
    names.sort();
    let next_cursor = (names.len() > page_size).then(|| (*names[page_size - 1]).clone());
    let mut response = json!({"namespaces": names.iter().take(page_size).map(|id| json!({"id":id})).collect::<Vec<_>>()});
    if let Some(cursor) = next_cursor {
        response["next_cursor"] = json!(cursor);
    }
    Ok(Json(response))
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
    if let Some(source) = object
        .get("branch_from_namespace")
        .or_else(|| object.get("copy_from_namespace"))
    {
        if object.len() != 1 {
            return Err(bad(
                "namespace copy cannot be combined with other write fields",
            ));
        }
        if source
            .as_object()
            .is_some_and(|config| config.len() != 1 || !config.contains_key("source_namespace"))
        {
            return Err(bad("only local source_namespace copies are supported"));
        }
        let source = source
            .as_str()
            .or_else(|| source.get("source_namespace").and_then(Value::as_str))
            .ok_or_else(|| bad("copy source_namespace is required"))?;
        namespace_name(source)?;
        if guard.contains_key(&name) {
            return Err(ApiError(
                StatusCode::CONFLICT,
                "destination namespace already exists".into(),
            ));
        }
        let namespace = guard.get(source).cloned().ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                "source namespace does not exist".into(),
            )
        })?;
        let rows = namespace.rows.len();
        persist_namespace(&state, &mut guard, name, namespace)?;
        return Ok(Json(
            json!({"status":"OK","message":"success","rows_affected":rows,"billing":{"billable_logical_bytes_written":0}}),
        ));
    }
    if !guard.contains_key(&name)
        && body
            .get("upsert_rows")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
        && body
            .get("upsert_columns")
            .and_then(|columns| columns.get("id"))
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
