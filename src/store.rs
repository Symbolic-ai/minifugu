use crate::embedding::EmbeddingMode;
use crate::query::{filter_matches, validate_filter};
use crate::text;
use crate::vector;
use base64::Engine;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Namespace {
    pub(crate) schema: Map<String, Value>,
    pub(crate) rows: BTreeMap<String, Map<String, Value>>,
    pub(crate) distance_metric: Option<String>,
    #[serde(default)]
    pub(crate) created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub(crate) updated_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub(crate) last_write_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub(crate) approx_logical_bytes: Option<usize>,
    #[serde(default)]
    pub(crate) read_only: bool,
}

impl Namespace {
    pub(crate) fn touch_write(&mut self) {
        let now = Utc::now();
        self.created_at.get_or_insert(now);
        self.updated_at = Some(now);
        self.last_write_at = Some(now);
        self.approx_logical_bytes = Some(self.compute_logical_bytes());
    }

    pub(crate) fn touch_schema(&mut self) {
        let now = Utc::now();
        self.created_at.get_or_insert(now);
        self.updated_at = Some(now);
    }

    pub(crate) fn touch_clone(&mut self) {
        self.last_write_at = self.last_write_at.or(self.updated_at);
        let now = Utc::now();
        self.created_at = Some(now);
        self.updated_at = Some(now);
        self.approx_logical_bytes = Some(self.compute_logical_bytes());
    }

    fn compute_logical_bytes(&self) -> usize {
        self.rows
            .values()
            .map(|row| serde_json::to_vec(row).map_or(0, |bytes| bytes.len()))
            .sum()
    }

    pub(crate) fn logical_bytes(&self) -> usize {
        self.approx_logical_bytes
            .unwrap_or_else(|| self.compute_logical_bytes())
    }

    /// The schema as the live service reports it. `GET .../schema` lists every attribute
    /// with `type`, `filterable` and `full_text_search`, using `null` where an option does
    /// not apply. Metadata leaves out null options and describes the vector indexes.
    pub(crate) fn schema_view(&self, view: SchemaView) -> Value {
        let mut attributes = Map::new();
        for (field, definition) in &self.schema {
            let kind = field_type(definition);
            let mut attribute = Map::new();
            attribute.insert("type".into(), json!(kind));
            let filterable = (field != "id" && !(field == "vector" && is_fixed_vector(definition)))
                .then(|| {
                    definition
                        .get("filterable")
                        .and_then(Value::as_bool)
                        .unwrap_or_else(|| default_filterable(definition))
                });
            let full_text_search = definition
                .get("full_text_search")
                .filter(|config| config.is_object() || **config == json!(true))
                .map(text::normalized_config);
            let fixed_vector = is_fixed_vector(definition);
            match view {
                SchemaView::Schema => {
                    attribute.insert("filterable".into(), json!(filterable));
                    attribute.insert("full_text_search".into(), json!(full_text_search));
                    if fixed_vector {
                        attribute.insert("ann".into(), json!(true));
                    }
                }
                SchemaView::Metadata => {
                    if let Some(filterable) = filterable {
                        attribute.insert("filterable".into(), json!(filterable));
                    }
                    if let Some(config) = full_text_search {
                        attribute.insert("full_text_search".into(), config);
                    }
                    if fixed_vector {
                        let metric = self.distance_metric.as_deref().unwrap_or("cosine_distance");
                        attribute.insert("ann".into(), json!({"distance_metric": metric}));
                    }
                    if let Some(sparse) = definition.get("sparse_knn") {
                        attribute.insert("sparse_knn".into(), sparse.clone());
                    }
                }
            }
            for option in ["regex", "glob", "fuzzy"] {
                if definition.get(option) == Some(&Value::Bool(true)) {
                    attribute.insert(option.into(), json!(true));
                }
            }
            if let Some(embed) = definition.get("embed") {
                let generated = format!("embed_{field}");
                if matches!(view, SchemaView::Metadata) {
                    attribute.insert(
                        "embed".into(),
                        json!({"attribute":generated,"model":embed["model"]}),
                    );
                }
                let kind = format!("[{}]f16", embed["dims"].as_u64().unwrap());
                let generated_attribute = match view {
                    SchemaView::Schema => json!({
                        "type":kind,"filterable":false,"full_text_search":null,"ann":true
                    }),
                    SchemaView::Metadata => json!({
                        "type":kind,"filterable":false,
                        "ann":{"distance_metric":self.distance_metric.as_deref().unwrap_or("cosine_distance")}
                    }),
                };
                attributes.insert(generated, generated_attribute);
            }
            attributes.insert(field.clone(), Value::Object(attribute));
        }
        Value::Object(attributes)
    }

    pub(crate) fn metadata(&self) -> Value {
        let created_at = self.created_at.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
        let updated_at = self.updated_at.unwrap_or(created_at);
        let mut metadata = json!({
            "schema": self.schema_view(SchemaView::Metadata),
            "approx_row_count": self.rows.len(),
            "approx_logical_bytes": self.logical_bytes(),
            "created_at": created_at,
            "updated_at": updated_at,
            "last_write_at": self.last_write_at.unwrap_or(updated_at).format("%Y-%m-%dT%H:%M:%S.000000000Z").to_string(),
            "encryption": {"sse": true},
            "index": {"status": "up-to-date"}
        });
        if self.read_only {
            metadata["read_only"] = json!(true);
        }
        metadata
    }

    pub(crate) fn recall(&self, body: &Value) -> Result<Value, String> {
        let options = body.as_object().ok_or("recall body must be an object")?;
        for key in options.keys() {
            if !matches!(
                key.as_str(),
                "num" | "top_k" | "filters" | "include_ground_truth"
            ) {
                return Err(format!("unsupported recall field {key}"));
            }
        }
        let num = options
            .get("num")
            .map_or(Some(25), Value::as_u64)
            .filter(|n| (1..=1000).contains(n))
            .ok_or("num must be between 1 and 1000")? as usize;
        let top_k = options
            .get("top_k")
            .map_or(Some(10), Value::as_u64)
            .filter(|n| (1..=1000).contains(n))
            .ok_or("top_k must be between 1 and 1000")?;
        if options
            .get("include_ground_truth")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err("include_ground_truth must be a boolean".into());
        }
        if let Some(filter) = options.get("filters") {
            validate_filter(filter, &self.schema)?;
        }
        let vector_field = self
            .schema
            .iter()
            .find(|(_, definition)| {
                vector::dimensions(definition).is_some()
                    && definition.get("ann") == Some(&Value::Bool(true))
            })
            .map(|(field, _)| field.clone())
            .or_else(|| {
                self.schema
                    .iter()
                    .find(|(_, definition)| has_embed(definition))
                    .map(|(field, _)| format!("embed_{field}"))
            })
            .ok_or("recall requires an ANN-enabled vector attribute")?;
        let mut ground_truth = Vec::new();
        let mut total = 0usize;
        let mut searched = 0usize;
        for vector in self
            .rows
            .values()
            .filter_map(|row| row.get(&vector_field).and_then(Value::as_array))
            .take(num)
        {
            searched += 1;
            let mut query = json!({"rank_by":[vector_field,"ANN",vector],"limit":top_k,"include_attributes":true});
            if let Some(filter) = options.get("filters") {
                query["filters"] = filter.clone();
            }
            let result = self.query(&query)?;
            let neighbors = result["rows"]
                .as_array()
                .ok_or("invalid recall query result")?;
            total += neighbors.len();
            if options.get("include_ground_truth") == Some(&Value::Bool(true)) {
                ground_truth.push(json!({"query_vector":vector,"nearest_neighbors":neighbors}));
            }
        }
        let average = if searched == 0 {
            0.0
        } else {
            total as f64 / searched as f64
        };
        let mut result =
            json!({"avg_recall":1.0,"avg_exhaustive_count":average,"avg_ann_count":average});
        if options.get("include_ground_truth") == Some(&Value::Bool(true)) {
            result["ground_truth"] = json!(ground_truth);
        }
        Ok(result)
    }
}

pub enum WriteError {
    Invalid(String),
    EmbeddingUnavailable,
}

impl From<String> for WriteError {
    fn from(value: String) -> Self {
        Self::Invalid(value)
    }
}

impl From<&str> for WriteError {
    fn from(value: &str) -> Self {
        Self::Invalid(value.to_owned())
    }
}

impl Namespace {
    pub async fn write(
        &mut self,
        body: &Value,
        embedder: &EmbeddingMode,
    ) -> Result<Value, WriteError> {
        crate::query::reset_text_caches();
        let object = body.as_object().ok_or("write body must be an object")?;
        validate_write_keys(object)?;
        if let Some(schema) = object.get("schema") {
            let schema = schema.as_object().ok_or("schema must be an object")?;
            for (field, definition) in schema {
                validate_definition(field, definition)?;
                let definition = if let Some(previous) = self.schema.get(field) {
                    if field_type(previous) != field_type(definition) {
                        return Err(format!("cannot change the type of attribute {field}").into());
                    }
                    if field == "id" {
                        definition.clone()
                    } else if let Some(update) = definition.as_object() {
                        let mut merged = previous.as_object().cloned().unwrap_or_else(|| {
                            Map::from_iter([("type".into(), json!(field_type(previous)))])
                        });
                        merged
                            .entry("filterable")
                            .or_insert_with(|| json!(effective_filterable(previous)));
                        let mut update = update.clone();
                        if let Some(new_fts) = update.get("full_text_search").cloned() {
                            let prior_fts = merged.get("full_text_search");
                            if new_fts == Value::Bool(true)
                                && prior_fts.is_some_and(Value::is_object)
                            {
                                update.remove("full_text_search");
                            } else if let (Some(old), Some(new)) =
                                (prior_fts.and_then(Value::as_object), new_fts.as_object())
                            {
                                let mut options = old.clone();
                                options.extend(new.clone());
                                update.insert("full_text_search".into(), Value::Object(options));
                            }
                        }
                        merged.extend(update);
                        Value::Object(merged)
                    } else {
                        // Shorthand replaces search/index options but leaves the
                        // current filterability of an existing field in place.
                        json!({"type":definition,"filterable":effective_filterable(previous)})
                    }
                } else {
                    definition.clone()
                };
                validate_definition(field, &definition)?;
                self.schema.insert(field.clone(), definition);
            }
        }
        let mut normalized = body.clone();
        vector::normalize_write(&mut normalized, &self.schema)?;
        let object = normalized
            .as_object()
            .ok_or("write body must be an object")?;
        validate_distinct_document_ids(object)?;
        for (flag, operation) in [
            ("delete_by_filter_allow_partial", "delete_by_filter"),
            ("patch_by_filter_allow_partial", "patch_by_filter"),
        ] {
            if let Some(value) = object.get(flag) {
                if !value.is_boolean() {
                    return Err(format!("{flag} must be a boolean").into());
                }
                if !object.contains_key(operation) {
                    return Err(format!("{flag} requires {operation}").into());
                }
            }
        }
        if object
            .get("disable_backpressure")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err("disable_backpressure must be a boolean".into());
        }
        if object.get("disable_backpressure") == Some(&Value::Bool(true))
            && !["upsert_rows", "upsert_columns", "deletes"]
                .iter()
                .any(|operation| object.contains_key(*operation))
        {
            return Err(
                "disable_backpressure is only supported for upserts and delete-by-id".into(),
            );
        }
        if let Some(metric) = object.get("distance_metric") {
            let metric = metric.as_str().ok_or("distance_metric must be a string")?;
            if !matches!(metric, "cosine_distance" | "euclidean_squared") {
                return Err(WriteError::Invalid(
                    "distance_metric must be cosine_distance or euclidean_squared".into(),
                ));
            }
            if let Some(existing) = self.distance_metric.as_deref() {
                if metric != existing {
                    return Err(format!(
                        "distance metric mismatch, expected {existing}, got {metric}"
                    )
                    .into());
                }
            }
            if !self.schema.values().any(|v| {
                vector::dimensions(v).is_some()
                    || vector::multi_dimensions(v).is_some()
                    || has_embed(v)
            }) {
                return Err(WriteError::Invalid(
                    "distance_metric requires a vector attribute".into(),
                ));
            }
            self.distance_metric = Some(metric.into());
        }
        if let Some(schema) = object.get("schema").and_then(Value::as_object) {
            for definition in schema.values() {
                if let Some(metric) = definition.pointer("/ann/distance_metric") {
                    if metric.is_null() {
                        continue;
                    }
                    let Some(namespace_metric) = self.distance_metric.as_deref() else {
                        return Err("distance_metric must be specified at the top level of the write request, not in ann".into());
                    };
                    if metric.as_str() != Some(namespace_metric) {
                        return Err(format!(
                            "distance metric mismatch, expected {namespace_metric}, got {metric}"
                        )
                        .into());
                    }
                }
            }
        }
        if self.distance_metric.is_none()
            && self.schema.values().any(|definition| {
                vector::dimensions(definition).is_some()
                    && vector::multi_dimensions(definition).is_none()
            })
        {
            return Err(
                "distance_metric must be specified for write to namespace with a vector".into(),
            );
        }
        for (condition, operations) in [
            ("upsert_condition", &["upsert_rows", "upsert_columns"][..]),
            ("patch_condition", &["patch_rows", "patch_columns"][..]),
            ("delete_condition", &["deletes"][..]),
        ] {
            if let Some(filter) = object.get(condition) {
                if !operations
                    .iter()
                    .any(|operation| object.contains_key(*operation))
                {
                    return Err(format!("{condition} requires a matching write operation").into());
                }
                validate_filter(filter, &self.schema)?;
            }
        }
        let mut upserted_ids = Vec::new();
        let mut patched_ids = Vec::new();
        let mut deleted_ids = Vec::new();
        if let Some(filter) = object.get("delete_by_filter") {
            validate_filter(filter, &self.schema)?;
            self.rows.retain(|_, row| {
                if filter_matches(filter, row, &self.schema) {
                    deleted_ids.push(row["id"].clone());
                    false
                } else {
                    true
                }
            });
            sort_ids(&mut deleted_ids);
        }
        if let Some(deletes) = object.get("deletes") {
            let deletes = deletes.as_array().ok_or("deletes must be an array")?;
            for id in deletes {
                let key = id_key(id)?;
                let current = self.rows.get(&key);
                // Plain delete-by-ID is an acknowledged write even when the row was
                // already absent. A conditional delete skips absent rows instead.
                if object.contains_key("delete_condition")
                    && (current.is_none()
                        || !condition_matches(object, "delete_condition", current, &self.schema))
                {
                    continue;
                }
                self.rows.remove(&key);
                deleted_ids.push(id.clone());
            }
        }
        if let Some(patch) = object.get("patch_by_filter") {
            let patch = patch
                .as_object()
                .ok_or("patch_by_filter must be an object")?;
            let filter = patch
                .get("filters")
                .ok_or("patch_by_filter requires filters")?;
            let values = patch
                .get("patch")
                .and_then(Value::as_object)
                .ok_or("patch_by_filter requires a patch object")?;
            if values.contains_key("id") {
                return Err("patch_by_filter cannot change id".into());
            }
            validate_filter(filter, &self.schema)?;
            self.validate_patch(values)?;
            for row in self.rows.values_mut() {
                if filter_matches(filter, row, &self.schema) {
                    for (field, value) in values {
                        if value.is_null() {
                            row.remove(field);
                        } else {
                            row.insert(field.clone(), value.clone());
                        }
                    }
                    patched_ids.push(row["id"].clone());
                }
            }
            sort_ids(&mut patched_ids);
        }
        if let Some(rows) = write_rows(object, "patch_rows", "patch_columns")? {
            for patch in &rows {
                let patch = patch
                    .as_object()
                    .ok_or("each patch row must be an object")?;
                let id = patch.get("id").ok_or("patch row requires id")?;
                validate_id(id, self.schema.get("id"))?;
                self.validate_patch(patch)?;
                let key = id_key(id)?;
                if !condition_matches(object, "patch_condition", self.rows.get(&key), &self.schema)
                {
                    continue;
                }
                if let Some(row) = self.rows.get_mut(&key) {
                    for (field, value) in patch {
                        if field != "id" {
                            if value.is_null() {
                                row.remove(field);
                            } else {
                                row.insert(field.clone(), value.clone());
                            }
                        }
                    }
                    patched_ids.push(id.clone());
                }
            }
        }
        if let Some(rows) = write_rows(object, "upsert_rows", "upsert_columns")? {
            for row in &rows {
                let mut row = row
                    .as_object()
                    .ok_or("each upsert row must be an object")?
                    .clone();
                let id = row.get("id").ok_or("upsert row requires id")?.clone();
                validate_id(&id, self.schema.get("id"))?;
                if !condition_matches(
                    object,
                    "upsert_condition",
                    self.rows.get(&id_key(&id)?),
                    &self.schema,
                ) {
                    continue;
                }
                for (field, definition) in &self.schema {
                    if is_fixed_vector(definition) && !row.contains_key(field) {
                        return Err(format!("upsert row requires vector attribute {field}").into());
                    }
                }
                for (field, value) in &row {
                    validate_attribute_name(field)?;
                    // `[]unknown` comes from an empty array; the first non-empty array
                    // settles the element type.
                    let unknown = self.schema.get(field) == Some(&json!("[]unknown"))
                        && value.as_array().is_some_and(|values| !values.is_empty());
                    if !self.schema.contains_key(field) || unknown {
                        match infer_type(field, value)? {
                            Some(inferred) => {
                                self.schema.insert(field.clone(), inferred);
                            }
                            None => continue,
                        }
                    }
                    validate_value(field, value, &self.schema[field])?;
                }
                for (field, definition) in &self.schema {
                    if has_embed(definition) {
                        if let Some(Value::String(text)) = row.get(field) {
                            let dims = definition
                                .get("embed")
                                .and_then(|e| e.get("dims"))
                                .and_then(Value::as_u64)
                                .unwrap_or(1536) as usize;
                            let model = definition
                                .get("embed")
                                .and_then(|e| e.get("model"))
                                .and_then(Value::as_str)
                                .unwrap_or("openai/text-embedding-3-small");
                            let vector = embedder
                                .embed(text, model, dims)
                                .await
                                .map_err(|_| WriteError::EmbeddingUnavailable)?;
                            row.insert(format!("embed_{field}"), json!(vector));
                        }
                    }
                }
                row.retain(|field, value| field == "id" || !value.is_null());
                self.rows.insert(id_key(&id)?, row);
                upserted_ids.push(id);
            }
        }
        let mut result = json!({
            "status":"OK", "message":"success",
            "rows_affected": upserted_ids.len() + patched_ids.len() + deleted_ids.len(),
            "rows_upserted": upserted_ids.len(),
            "rows_patched": patched_ids.len(),
            "rows_deleted": deleted_ids.len(),
            "billing":{"billable_logical_bytes_written":0}
        });
        if object.get("return_affected_ids") == Some(&Value::Bool(true)) {
            if !upserted_ids.is_empty() {
                result["upserted_ids"] = json!(upserted_ids);
            }
            if !patched_ids.is_empty() {
                result["patched_ids"] = json!(patched_ids);
            }
            if !deleted_ids.is_empty() {
                result["deleted_ids"] = json!(deleted_ids);
            }
        }
        Ok(result)
    }

    fn validate_patch(&self, values: &Map<String, Value>) -> Result<(), WriteError> {
        for (field, value) in values {
            validate_attribute_name(field)?;
            if field == "id" {
                continue;
            }
            let definition = self
                .schema
                .get(field)
                .ok_or_else(|| format!("attribute {field} does not exist in schema"))?;
            if has_embed(definition) {
                return Err(format!("patching embedded attribute {field} is unsupported").into());
            }
            if vector::dimensions(definition).is_some()
                || vector::multi_dimensions(definition).is_some()
            {
                return Err("💔 patching vectors is currently unsupported".into());
            }
            validate_value(field, value, definition)?;
        }
        Ok(())
    }
}

fn validate_distinct_document_ids(object: &Map<String, Value>) -> Result<(), WriteError> {
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = 0;
    if let Some(deletes) = object.get("deletes") {
        for id in deletes.as_array().ok_or("deletes must be an array")? {
            if !seen.insert(id_key(id)?) {
                duplicates += 1;
            }
        }
    }
    for (rows, columns) in [
        ("patch_rows", "patch_columns"),
        ("upsert_rows", "upsert_columns"),
    ] {
        if let Some(documents) = write_rows(object, rows, columns)? {
            for document in documents {
                let id = document.get("id").ok_or("write row requires id")?;
                if !seen.insert(id_key(id)?) {
                    duplicates += 1;
                }
            }
        }
    }
    if duplicates > 0 {
        return Err(format!("💔 This upsert contains {duplicates} duplicate document IDs and was not written. You should ensure that individual upserts do not include duplicate documents.").into());
    }
    Ok(())
}

fn condition_matches(
    object: &Map<String, Value>,
    condition: &str,
    current: Option<&Map<String, Value>>,
    schema: &Map<String, Value>,
) -> bool {
    object
        .get(condition)
        .is_none_or(|filter| filter_matches(filter, current.unwrap_or(&Map::new()), schema))
}

pub(crate) fn validate_write_keys(object: &Map<String, Value>) -> Result<(), String> {
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "schema"
                | "distance_metric"
                | "delete_by_filter"
                | "deletes"
                | "patch_by_filter"
                | "patch_rows"
                | "patch_columns"
                | "upsert_rows"
                | "upsert_columns"
                | "return_affected_ids"
                | "upsert_condition"
                | "patch_condition"
                | "delete_condition"
                | "branch_from_namespace"
                | "copy_from_namespace"
                | "delete_by_filter_allow_partial"
                | "patch_by_filter_allow_partial"
                | "disable_backpressure"
        ) {
            return Err(format!("unsupported write field {key}"));
        }
    }
    Ok(())
}

fn write_rows(
    object: &Map<String, Value>,
    row_key: &str,
    column_key: &str,
) -> Result<Option<Vec<Value>>, WriteError> {
    match (object.get(row_key), object.get(column_key)) {
        (Some(_), Some(_)) => Err(format!("{row_key} and {column_key} cannot be combined").into()),
        (Some(rows), None) => Ok(Some(
            rows.as_array()
                .ok_or_else(|| format!("{row_key} must be an array"))?
                .clone(),
        )),
        (None, Some(columns)) => Ok(Some(rows_from_columns(columns)?)),
        (None, None) => Ok(None),
    }
}

fn rows_from_columns(columns: &Value) -> Result<Vec<Value>, WriteError> {
    let columns = columns.as_object().ok_or("columns must be an object")?;
    let ids = columns
        .get("id")
        .and_then(Value::as_array)
        .ok_or("columns require an id array")?;
    let mut rows = vec![Map::new(); ids.len()];
    for (field, values) in columns {
        let values = values
            .as_array()
            .ok_or_else(|| format!("column {field} must be an array"))?;
        if values.len() != rows.len() {
            return Err(format!("column {field} length must match id length").into());
        }
        for (row, value) in rows.iter_mut().zip(values) {
            row.insert(field.clone(), value.clone());
        }
    }
    Ok(rows.into_iter().map(Value::Object).collect())
}

#[derive(Clone, Copy)]
pub(crate) enum SchemaView {
    Schema,
    Metadata,
}

/// Indexed text, fixed vectors, sparse vectors and bytes are not filterable unless the
/// schema says so; every other attribute is.
fn default_filterable(definition: &Value) -> bool {
    let enabled = |option: &str| {
        definition
            .get(option)
            .is_some_and(|value| value == &json!(true) || value.is_object())
    };
    !(enabled("full_text_search")
        || enabled("regex")
        || enabled("glob")
        || enabled("fuzzy")
        || is_fixed_vector(definition)
        || matches!(field_type(definition), "{}f16" | "bytes"))
}

fn effective_filterable(definition: &Value) -> bool {
    definition
        .get("filterable")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| default_filterable(definition))
}

fn is_fixed_vector(definition: &Value) -> bool {
    vector::dimensions(definition).is_some() && vector::multi_dimensions(definition).is_none()
}

pub(crate) fn field_type(definition: &Value) -> &str {
    definition
        .as_str()
        .or_else(|| definition.get("type").and_then(Value::as_str))
        .unwrap_or("")
}

pub(crate) fn has_embed(definition: &Value) -> bool {
    definition.get("embed").is_some()
}

pub(crate) fn known_field(schema: &Map<String, Value>, field: &str) -> bool {
    schema.contains_key(field)
        || field
            .strip_prefix("embed_")
            .is_some_and(|base| schema.get(base).is_some_and(has_embed))
}

fn validate_attribute_name(field: &str) -> Result<(), String> {
    if field.is_empty() {
        return Err("attribute name cannot be empty".into());
    }
    if field.len() > 128 {
        return Err(format!("attribute name is too long: {field}"));
    }
    if field.starts_with('$') {
        return Err(format!("cannot use reserved attribute name {field}"));
    }
    Ok(())
}

fn validate_definition(field: &str, definition: &Value) -> Result<(), String> {
    validate_attribute_name(field)?;
    if definition
        .as_object()
        .is_some_and(|config| !config.contains_key("type"))
    {
        return Err(crate::shape_error(format!(
            "schema.{field} requires a type"
        )));
    }
    if let Some(config) = definition.as_object() {
        for (option, value) in config {
            match option.as_str() {
                "type" if !value.is_string() => {
                    return Err(format!(
                        "schema option type must be a string for attribute {field}"
                    ));
                }
                "filterable" | "regex" | "glob" | "full_text_search" | "fuzzy"
                    if !value.is_boolean()
                        && !(option == "full_text_search" && value.is_object()) =>
                {
                    return Err(format!(
                        "schema option {option} must be a boolean for attribute {field}"
                    ));
                }
                "ann" if !value.is_boolean() && !value.is_object() => {
                    return Err(format!("invalid ann configuration for attribute {field}"));
                }
                "ann"
                    if value == &Value::Bool(true)
                        && vector::dimensions(definition).is_none()
                        && vector::multi_dimensions(definition).is_none() =>
                {
                    return Err(format!("ann requires a vector attribute: {field}"));
                }
                "full_text_search" if value.is_object() => {
                    if !matches!(field_type(definition), "string" | "[]string") {
                        return Err(format!(
                            "full_text_search requires a string or []string attribute: {field}"
                        ));
                    }
                    text::validate_config(field_type(definition), value).map_err(|error| {
                        match error {
                            text::ConfigError::Shape(message) => {
                                crate::shape_error(format!("schema.{field}: {message}"))
                            }
                            text::ConfigError::Invalid(message) => message,
                        }
                    })?;
                }
                "ann" if value.is_object() => {
                    let ann = value.as_object().unwrap();
                    if vector::multi_dimensions(definition).is_some() {
                        if ann.len() != 1 || ann.get("late_interaction") != Some(&Value::Bool(true))
                        {
                            return Err(format!(
                                "invalid late-interaction configuration for attribute {field}"
                            ));
                        }
                    } else if vector::dimensions(definition).is_some() {
                        if ann.contains_key("late_interaction") {
                            return Err(format!("invalid ann configuration for attribute {field}"));
                        }
                        if ann.get("distance_metric").is_some_and(|metric| {
                            !metric.is_null()
                                && !matches!(
                                    metric.as_str(),
                                    Some("cosine_distance" | "euclidean_squared")
                                )
                        }) {
                            return Err(crate::shape_error(format!(
                                "schema.{field}.ann.distance_metric is invalid"
                            )));
                        }
                    } else {
                        return Err(format!("invalid ann configuration for attribute {field}"));
                    }
                }
                "sparse_knn" => {
                    if field_type(definition) != "{}f16"
                        || value.as_object().is_none_or(|config| {
                            config.len() != 1
                                || config.get("distance_metric") != Some(&json!("dot_product"))
                        })
                    {
                        return Err(format!(
                            "invalid sparse_knn configuration for attribute {field}"
                        ));
                    }
                }
                "embed" if !value.is_object() => {
                    return Err(format!(
                        "schema option embed must be an object for attribute {field}"
                    ));
                }
                "embed" => {
                    if value.as_object().is_some_and(|embed| {
                        embed
                            .keys()
                            .any(|key| !matches!(key.as_str(), "model" | "dims"))
                    }) {
                        return Err(format!("unsupported embed option for attribute {field}"));
                    }
                }
                "type" | "filterable" | "ann" | "regex" | "glob" | "full_text_search" | "fuzzy" => {
                }
                _ => {
                    return Err(format!(
                        "unsupported schema option {option} for attribute {field}"
                    ))
                }
            }
        }
    }
    if let Some(embed) = definition.get("embed") {
        if !embed.get("model").is_some_and(Value::is_string)
            || !embed
                .get("dims")
                .and_then(Value::as_u64)
                .is_some_and(|dims| dims > 0 && dims <= 3072)
        {
            return Err(format!("invalid embed configuration for attribute {field}"));
        }
    }
    if vector::dimensions(definition).is_some()
        && vector::multi_dimensions(definition).is_none()
        && !definition
            .get("ann")
            .is_some_and(|ann| ann == &json!(true) || ann.is_object())
        && definition.get("embed").is_none()
    {
        return Err(format!("vector attribute '{field}' must have ann:true"));
    }
    if matches!(field_type(definition), "bytes" | "{}f16")
        && definition.get("filterable") == Some(&Value::Bool(true))
    {
        return Err(format!(
            "{} attribute {field} cannot be filterable",
            field_type(definition)
        ));
    }
    match field_type(definition) {
        "uuid" | "uint" | "int" | "float" | "string" | "bool" | "datetime" | "bytes" | "{}f16"
        | "[]unknown" | "[]uuid" | "[]uint" | "[]int" | "[]float" | "[]string" | "[]bool"
        | "[]datetime" => Ok(()),
        _ if vector::multi_dimensions(definition).is_some_and(|n| n > 0 && n <= 3072) => Ok(()),
        value
            if value.starts_with('[')
                && (value.ends_with("]f16")
                    || value.ends_with("]f32")
                    || value.ends_with("]i8"))
                && value
                    .trim_start_matches('[')
                    .split(']')
                    .next()
                    .and_then(|n| n.parse::<usize>().ok())
                    .is_some_and(|n| n > 0 && n <= 3072) =>
        {
            Ok(())
        }
        _ => Err(format!("unsupported schema type for attribute {field}")),
    }
}

fn sort_ids(ids: &mut [Value]) {
    ids.sort_by(|left, right| match (left.as_u64(), right.as_u64()) {
        (Some(left), Some(right)) => left.cmp(&right),
        _ => left.as_str().cmp(&right.as_str()),
    });
}

fn validate_id(value: &Value, definition: Option<&Value>) -> Result<(), String> {
    id_key(value)?;
    if let Some(definition) = definition {
        validate_value("id", value, definition)?;
    }
    Ok(())
}

fn id_key(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) if s.len() <= 64 => Ok(format!("s:{s}")),
        Value::String(s) => Err(format!(
            "id string must be at most 64 bytes, got {}",
            s.len()
        )),
        Value::Number(n) if n.as_u64().is_some() => Ok(format!("n:{n}")),
        _ => Err("id must be a string or unsigned integer".into()),
    }
}

/// Infers an attribute type from its first value with the live service's rules: numbers are
/// `int`, arrays take their element type, arrays of arrays are multi-vectors and objects are
/// sparse vectors. `None` means the value is null and adds no schema entry.
fn infer_type(field: &str, value: &Value) -> Result<Option<Value>, String> {
    let incompatible = |inferred: &str, detail: String| {
        format!(
            "inferred schema type for attribute '{field}' as {inferred}, but got an incompatible value ({detail}). consider specifying the explicit type you want for this attribute in the schema"
        )
    };
    let name = match value {
        Value::Null => return Ok(None),
        Value::Number(_) if field == "id" => "uint".to_owned(),
        Value::String(_) => "string".to_owned(),
        Value::Bool(_) => "bool".to_owned(),
        Value::Number(number) if number.is_i64() => "int".to_owned(),
        Value::Number(_) => {
            return Err(incompatible(
                "int",
                "of type number, number cannot be represented as signed 64-bit integer".into(),
            ))
        }
        Value::Object(_) => "{}f16".to_owned(),
        Value::Array(values) => match values.first() {
            None => "[]unknown".to_owned(),
            Some(Value::Bool(_)) => "[]bool".to_owned(),
            Some(Value::String(_)) => "[]string".to_owned(),
            Some(Value::Number(_)) => {
                if let Some(index) = values.iter().position(|value| !value.is_i64()) {
                    return Err(incompatible(
                        "int",
                        format!("of type number, number at index {index} cannot be represented as signed 64-bit integer"),
                    ));
                }
                "[]int".to_owned()
            }
            Some(Value::Array(first)) => {
                let dimensions = first.len();
                if let Some((index, vector)) = values.iter().enumerate().find(|(_, vector)| {
                    vector
                        .as_array()
                        .is_none_or(|vector| vector.len() != dimensions)
                }) {
                    let actual = vector.as_array().map_or(0, Vec::len);
                    return Err(incompatible(
                        &format!("vector with {dimensions} dimensions"),
                        format!("of type vector with {actual} dimensions, vector at index {index} has {actual} dimensions, expected {dimensions}"),
                    ));
                }
                format!("[][{dimensions}]f32")
            }
            Some(_) => return Err("cannot infer schema from this array".into()),
        },
    };
    Ok(Some(json!(name)))
}

fn validate_value(field: &str, value: &Value, definition: &Value) -> Result<(), String> {
    if value.is_null() && field != "id" {
        return Ok(());
    }
    let valid = match field_type(definition) {
        "[]unknown" => value.as_array().is_some_and(Vec::is_empty),
        "uuid" => value.as_str().is_some_and(uuid_like),
        "uint" => value.as_u64().is_some(),
        "int" => value.as_i64().is_some(),
        "float" => value.is_number(),
        "string" | "datetime" => value.is_string(),
        "bool" => value.is_boolean(),
        "bytes" => value.as_str().is_some_and(|encoded| {
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .is_ok_and(|bytes| bytes.len() <= 8 * 1024 * 1024)
        }),
        "[]uuid" => value
            .as_array()
            .is_some_and(|a| a.iter().all(|v| v.as_str().is_some_and(uuid_like))),
        "[]uint" => value
            .as_array()
            .is_some_and(|a| a.iter().all(|v| v.as_u64().is_some())),
        "[]int" => value
            .as_array()
            .is_some_and(|a| a.iter().all(|v| v.as_i64().is_some())),
        "[]float" => value
            .as_array()
            .is_some_and(|a| a.iter().all(Value::is_number)),
        "[]string" | "[]datetime" => value
            .as_array()
            .is_some_and(|a| a.iter().all(Value::is_string)),
        "[]bool" => value
            .as_array()
            .is_some_and(|a| a.iter().all(Value::is_boolean)),
        "{}f16" => value.as_object().is_some_and(|sparse| {
            sparse.len() <= 1024
                && sparse.keys().all(|key| !key.is_empty())
                && sparse.values().all(Value::is_number)
        }),
        _ if vector::multi_dimensions(definition).is_some() => vector::multi_dimensions(definition)
            .is_some_and(|dimensions| {
                value.as_array().is_some_and(|vectors| {
                    vector::multi_vector_within_limit(vectors.len(), dimensions)
                        && vectors.iter().all(|vector| {
                            vector.as_array().is_some_and(|elements| {
                                elements.len() == dimensions
                                    && elements.iter().all(Value::is_number)
                            })
                        })
                })
            }),
        vector if vector::dimensions(definition).is_some() => {
            let expected = vector
                .trim_start_matches('[')
                .split(']')
                .next()
                .and_then(|n| n.parse::<usize>().ok());
            value.as_array().is_some_and(|array| {
                Some(array.len()) == expected
                    && array.iter().all(|item| {
                        if vector.ends_with("]i8") {
                            item.as_f64().is_some_and(|number| {
                                number.fract() == 0.0 && (-128.0..=127.0).contains(&number)
                            })
                        } else {
                            item.is_number()
                        }
                    })
            })
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("invalid value for attribute {field}"))
    }
}

fn uuid_like(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes.iter().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                *b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}
