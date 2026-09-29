#![forbid(unsafe_code)]

mod embedding;
mod highlight;
mod persistence;
mod query;
mod store;
mod text;
mod vector;

use axum::{
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    path::{Path as FilePath, PathBuf},
    sync::Arc,
};
use tokio::sync::RwLock;

pub use embedding::{deterministic_embedding, EmbeddingMode};
pub use store::Namespace;
/// Turbopuffer's documented maximum upsert request size. Axum's 2 MB default would reject
/// ordinary vector batches before validation.
const MAX_REQUEST_BYTES: usize = 512 * 1024 * 1024;

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

/// Marks a validation message as a JSON shape error. The live service reports those as HTTP
/// 422 from its request deserializer, and semantic errors as HTTP 400.
const SHAPE_ERROR: &str = "\u{1}shape:";

/// A validation message that becomes an HTTP 422 shape error.
pub(crate) fn shape_error(message: impl std::fmt::Display) -> String {
    format!("{SHAPE_ERROR}{message}")
}

fn bad(message: impl Into<String>) -> ApiError {
    let message = message.into();
    match message.strip_prefix(SHAPE_ERROR) {
        Some(detail) => ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("Failed to deserialize the JSON body into the target type: {detail}"),
        ),
        None => ApiError(StatusCode::BAD_REQUEST, message),
    }
}

/// A JSON request body. Parse failures use the API error shape, as they do live, instead of
/// the extractor's plain-text rejection.
struct JsonBody(Value);

impl<S: Send + Sync> axum::extract::FromRequest<S> for JsonBody {
    type Rejection = ApiError;

    async fn from_request(request: axum::extract::Request, state: &S) -> Result<Self, ApiError> {
        let bytes = axum::body::Bytes::from_request(request, state)
            .await
            .map_err(|rejection| ApiError(rejection.status(), rejection.body_text()))?;
        serde_json::from_slice(&bytes)
            .map(JsonBody)
            .map_err(|error| bad(format!("Failed to parse the request body as JSON: {error}")))
    }
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
        .route(
            "/v1/namespaces/{namespace}/metadata",
            get(get_metadata).patch(update_metadata),
        )
        .route("/v2/namespaces/{namespace}/metadata", get(get_metadata))
        .route(
            "/v1/namespaces/{namespace}/hint_cache_warm",
            get(hint_cache_warm),
        )
        .route("/v1/namespaces/{namespace}/_debug/recall", post(recall))
        .route(
            "/v2/namespaces/{namespace}",
            post(write).delete(delete_namespace),
        )
        .route("/v2/namespaces/{namespace}/query", post(query_namespace))
        .route(
            "/v2/namespaces/{namespace}/explain_query",
            post(explain_query),
        )
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
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
    let cursor = params
        .get("cursor")
        .map(|cursor| decode_namespace_cursor(cursor))
        .transpose()?;
    let guard = state.namespaces.read().await;
    let mut names = guard
        .keys()
        .filter(|name| {
            params
                .get("prefix")
                .is_none_or(|prefix| name.starts_with(prefix))
        })
        .filter(|name| cursor.as_ref().is_none_or(|cursor| *name > cursor))
        .collect::<Vec<_>>();
    names.sort();
    let next_cursor = (names.len() >= page_size).then(|| {
        let last = names[page_size - 1];
        STANDARD.encode(
            serde_json::to_vec(&json!({
                "continuation_token": null,
                "start_after": format!("{last}-table/")
            }))
            .unwrap(),
        )
    });
    let mut response = json!({"namespaces": names.iter().take(page_size).map(|id| json!({"id":id})).collect::<Vec<_>>()});
    response["next_cursor"] = json!(next_cursor);
    Ok(Json(response))
}

fn decode_namespace_cursor(encoded: &str) -> Result<String, ApiError> {
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| bad("invalid cursor"))?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|_| bad("invalid cursor"))?;
    let object = value.as_object().ok_or_else(|| bad("invalid cursor"))?;
    if object.len() != 2 || object.get("continuation_token") != Some(&Value::Null) {
        return Err(bad("invalid cursor"));
    }
    object
        .get("start_after")
        .and_then(Value::as_str)
        .and_then(|start| start.strip_suffix("-table/"))
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| bad("invalid cursor"))
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
    Ok(Json(namespace.schema_view(store::SchemaView::Schema)))
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
    Ok(Json(namespace.metadata()))
}

async fn update_metadata(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(body): JsonBody,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let options = body
        .as_object()
        .ok_or_else(|| bad(shape_error("metadata body must be an object")))?;
    if options.get("pinning").is_some_and(|value| !value.is_null()) {
        return Err(bad("namespace pinning is unavailable locally"));
    }
    // Turbopuffer ignores unknown metadata keys. Keep that behavior so a client
    // can send harmless newer settings alongside read_only.
    let read_only = options
        .get("read_only")
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| bad(shape_error("read_only must be a boolean")))
        })
        .transpose()?;
    let mut guard = state.namespaces.write().await;
    let mut namespace = guard
        .get(&name)
        .cloned()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    if let Some(read_only) = read_only {
        namespace.read_only = read_only;
    }
    let response = namespace.metadata();
    persist_namespace(&state, &mut guard, name, namespace)?;
    Ok(Json(response))
}

async fn hint_cache_warm(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    if !state.namespaces.read().await.contains_key(&name) {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "namespace does not exist".into(),
        ));
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "status":"ACCEPTED", "message":"cache warm hint accepted"
        })),
    ))
}

async fn recall(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(body): JsonBody,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let guard = state.namespaces.read().await;
    let namespace = guard
        .get(&name)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    namespace.recall(&body).map(Json).map_err(bad)
}

async fn explain_query(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(body): JsonBody,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let guard = state.namespaces.read().await;
    let namespace = guard
        .get(&name)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    namespace.query(&body).map_err(bad)?;
    let plan = if let Some(queries) = body.get("queries").and_then(Value::as_array) {
        format!(
            "MiniFugu exact scan of {} rows for {} subqueries; local fusion",
            namespace.rows.len(),
            queries.len()
        )
    } else {
        let rank = if body.get("rank_by").is_some() {
            "ranked"
        } else {
            "unranked"
        };
        let filter = if body.get("filters").is_some() {
            "filtered"
        } else {
            "unfiltered"
        };
        format!(
            "MiniFugu exact {filter} {rank} scan of {} rows",
            namespace.rows.len()
        )
    };
    Ok(Json(json!({"plan_text":plan})))
}

async fn update_schema(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(schema): JsonBody,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let mut guard = state.namespaces.write().await;
    let mut namespace = guard
        .get(&name)
        .cloned()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "namespace does not exist".into()))?;
    if namespace.read_only {
        return Err(bad("💔 Writes not permitted. This namespace is read-only."));
    }
    namespace
        .write(&json!({"schema":schema}), &state.embedding)
        .await
        .map_err(|error| match error {
            store::WriteError::Invalid(message) => bad(message),
            store::WriteError::EmbeddingUnavailable => bad("embedding provider unavailable"),
        })?;
    namespace.touch_schema();
    let response = namespace.schema_view(store::SchemaView::Schema);
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
    JsonBody(body): JsonBody,
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
        let mut namespace = guard.get(source).cloned().ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                "source namespace does not exist".into(),
            )
        })?;
        namespace.touch_clone();
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
        && !object.contains_key("upsert_columns")
    {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "namespace does not exist".into(),
        ));
    }
    let mut namespace = guard.get(&name).cloned().unwrap_or_default();
    if namespace.read_only {
        return Err(bad("💔 Writes not permitted. This namespace is read-only."));
    }
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
    namespace.touch_write();
    persist_namespace(&state, &mut guard, name, namespace)?;
    Ok(Json(result))
}

async fn query_namespace(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(body): JsonBody,
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
