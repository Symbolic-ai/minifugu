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
use std::{collections::HashMap, path::Path as FilePath, sync::Arc};
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
    /// Only touched while `namespaces` is write-locked, so it never waits on itself.
    store: Option<std::sync::Mutex<persistence::Store>>,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(json!({"status":"error", "error":self.1}))).into_response()
    }
}

/// Regions the live service accepts as a `copy_from_namespace` `source_region`.
const COPY_REGIONS: &[&str] = &[
    "aws-ap-south-1",
    "aws-ap-southeast-2",
    "aws-ca-central-1",
    "aws-eu-central-1",
    "aws-eu-west-1",
    "aws-eu-west-2",
    "aws-us-east-1",
    "aws-us-east-2",
    "aws-us-west-2",
    "gcp-asia-northeast3",
    "gcp-asia-southeast1",
    "gcp-europe-west1",
    "gcp-europe-west3",
    "gcp-northamerica-northeast2",
    "gcp-us-central1",
    "gcp-us-east1",
    "gcp-us-east4",
    "gcp-us-west1",
];

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
    let (store, namespaces) = persistence::open(directory)?;
    Ok(router_with_state(embedding, Some(store), namespaces))
}

fn router_with_state(
    embedding: EmbeddingMode,
    store: Option<persistence::Store>,
    namespaces: HashMap<String, Namespace>,
) -> Router {
    let state = Arc::new(AppState {
        namespaces: RwLock::new(namespaces),
        embedding,
        store: store.map(std::sync::Mutex::new),
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
        // Keep the live STANDARD encoding. Valid namespace names and the fixed JSON
        // envelope are ASCII bytes that cannot produce '+' or '/' in base64.
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
    let Some(store) = &state.store else {
        guard.insert(name, namespace);
        return Ok(());
    };
    let mut store = store.lock().map_err(|_| persist_failed())?;
    store
        .put(&name, guard.get(&name), &namespace)
        .map_err(|_| persist_failed())?;
    guard.insert(name, namespace);
    compact(&mut store, guard);
    Ok(())
}

fn persist_failed() -> ApiError {
    ApiError(
        StatusCode::INTERNAL_SERVER_ERROR,
        "failed to persist namespace".into(),
    )
}

/// The write is already durable in the log, so a failed compaction is reported and
/// retried after a later write rather than failing this request.
fn compact(store: &mut persistence::Store, namespaces: &HashMap<String, Namespace>) {
    if let Err(error) = store.maybe_compact(namespaces) {
        eprintln!("MiniFugu could not compact its snapshot: {error}");
    }
}

async fn write(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(mut body): JsonBody,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    store::normalize_write_body(&mut body).map_err(bad)?;
    let object = body
        .as_object()
        .expect("normalized write body is an object");
    store::validate_write_conditions(object).map_err(bad)?;
    let mut guard = state.namespaces.write().await;
    if !store::has_write_operations(object, guard.contains_key(&name)) {
        return Err(bad("💔 no writes provided"));
    }
    let clone_kind = ["branch_from_namespace", "copy_from_namespace"]
        .into_iter()
        .find(|key| object.contains_key(*key));
    if let Some(kind) = clone_kind {
        let source = &object[kind];
        // A branch takes no other field; a copy may carry a destination `encryption`.
        let allowed =
            |key: &str| key == kind || (kind == "copy_from_namespace" && key == "encryption");
        if object.keys().any(|key| !allowed(key)) {
            return Err(bad(format!(
                "💔 {kind} cannot be used with other write request fields"
            )));
        }
        // MiniFugu is one keyless region, so a source in any live region and with any
        // source API key resolves locally. Other config keys are ignored, as they are live.
        if let Some(region) = source.get("source_region") {
            let region = region.as_str().ok_or_else(|| {
                bad(shape_error(
                    "copy_from_namespace.source_region must be a string",
                ))
            })?;
            if !COPY_REGIONS.contains(&region) {
                return Err(bad(format!(
                    "💔 region '{region}' is not available for cross-region copy_from_namespace. available regions: {}",
                    COPY_REGIONS.join(", ")
                )));
            }
        }
        if source
            .get("source_api_key")
            .is_some_and(|key| !key.is_string())
        {
            return Err(bad(shape_error(
                "copy_from_namespace.source_api_key must be a string",
            )));
        }
        let source = source
            .as_str()
            .or_else(|| source.get("source_namespace").and_then(Value::as_str))
            .ok_or_else(|| {
                bad(shape_error(
                    "copy_from_namespace: data did not match any variant of CopyFromNamespaceParams",
                ))
            })?;
        namespace_name(source)?;
        // A destination `encryption` replaces the key; without one the copy keeps the
        // source's key.
        let encryption = object
            .get("encryption")
            .map(store::cmek_key_name)
            .transpose()
            .map_err(bad)?;
        if source == name {
            return Err(bad("💔 Source and destination namespace can't be the same"));
        }
        if guard.contains_key(&name) {
            return Err(bad(format!(
                "💔 Destination namespace `{name}` already exists"
            )));
        }
        let mut namespace = guard.get(source).cloned().ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                format!("🤷 namespace '{source}' was not found"),
            )
        })?;
        namespace.touch_clone();
        if let Some(cmek_key_name) = encryption {
            namespace.cmek_key_name = cmek_key_name;
        }
        // Live reports a branch as affecting no rows; a copy reports the rows it copied.
        let (message, rows) = if kind == "branch_from_namespace" {
            ("namespace branch successful", 0)
        } else {
            ("namespace cloned successfully", namespace.rows.len())
        };
        persist_namespace(&state, &mut guard, name, namespace)?;
        return Ok(Json(
            json!({"status":"OK","message":message,"rows_affected":rows,"billing":{"billable_logical_bytes_written":0}}),
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
    // A new namespace takes its customer-managed key from the creating write. On an
    // existing namespace, live accepts its current setting; whether live applies a
    // different key cannot be checked without a cloud KMS, so MiniFugu rejects it
    // rather than drop it silently.
    if let Some(encryption) = object.get("encryption") {
        let key_name = store::cmek_key_name(encryption).map_err(bad)?;
        if !guard.contains_key(&name) {
            namespace.cmek_key_name = key_name;
        } else if key_name != namespace.cmek_key_name {
            return Err(bad(
                "encryption cannot be changed on an existing namespace in MiniFugu",
            ));
        }
    }
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
    namespace.touch_write(guard.get(&name));
    persist_namespace(&state, &mut guard, name, namespace)?;
    Ok(Json(result))
}

async fn query_namespace(
    Path(name): Path<String>,
    State(state): State<Shared>,
    headers: HeaderMap,
    JsonBody(mut body): JsonBody,
) -> Result<Json<Value>, ApiError> {
    authorized(&headers)?;
    namespace_name(&name)?;
    let requests = {
        let guard = state.namespaces.read().await;
        let namespace = guard.get(&name).ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                format!("namespace {name} does not exist"),
            )
        })?;
        let requests = embedding::prepare_query(&mut body, &namespace.schema).map_err(bad)?;
        if !requests.is_empty() {
            namespace.validate_body(&body).map_err(bad)?;
        }
        requests
    };
    embedding::materialize_query(&mut body, requests, &state.embedding)
        .await
        .map_err(|error| match error {
            embedding::QueryEmbeddingError::Invalid(message) => bad(message),
            embedding::QueryEmbeddingError::Unavailable => ApiError(
                StatusCode::BAD_GATEWAY,
                "embedding provider unavailable".into(),
            ),
        })?;
    // A schema update during the provider call is checked by the query against
    // the current namespace below; stale vector dimensions produce a clean 400.
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
    if let Some(store) = &state.store {
        let mut store = store.lock().map_err(|_| persist_failed())?;
        store.drop_namespace(&name).map_err(|_| persist_failed())?;
        guard.remove(&name);
        compact(&mut store, &guard);
    } else {
        guard.remove(&name);
    }
    Ok(Json(json!({"status":"ok"})))
}
