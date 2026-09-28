use crate::embedding::EmbeddingMode;
use crate::query::{filter_matches, validate_filter};
use crate::vector;
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
}

impl Namespace {
    pub(crate) fn touch(&mut self) {
        let now = Utc::now();
        self.created_at.get_or_insert(now);
        self.updated_at = Some(now);
    }

    pub(crate) fn metadata(&self) -> Value {
        let created_at = self.created_at.unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
        let updated_at = self.updated_at.unwrap_or(created_at);
        let approx_logical_bytes = self
            .rows
            .values()
            .map(|row| serde_json::to_vec(row).map_or(0, |bytes| bytes.len()))
            .sum::<usize>();
        json!({
            "schema": self.schema,
            "approx_row_count": self.rows.len(),
            "approx_logical_bytes": approx_logical_bytes,
            "created_at": created_at,
            "updated_at": updated_at,
            "encryption": {"sse": true},
            "index": {"status": "up-to-date"}
        })
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
                let kind = field_type(definition);
                kind.ends_with("]f16") || kind.ends_with("]f32") || kind.ends_with("]i8")
            })
            .map(|(field, _)| field.as_str())
            .ok_or("recall requires a vector attribute")?;
        let mut ground_truth = Vec::new();
        let mut total = 0usize;
        let mut searched = 0usize;
        for vector in self
            .rows
            .values()
            .filter_map(|row| row.get(vector_field).and_then(Value::as_array))
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
        let object = body.as_object().ok_or("write body must be an object")?;
        validate_write_keys(object)?;
        if let Some(schema) = object.get("schema") {
            let schema = schema.as_object().ok_or("schema must be an object")?;
            for (field, definition) in schema {
                validate_definition(field, definition)?;
                if let Some(previous) = self.schema.get(field) {
                    if field_type(previous) != field_type(definition) {
                        return Err(format!("cannot change the type of attribute {field}").into());
                    }
                }
                self.schema.insert(field.clone(), definition.clone());
            }
        }
        let mut normalized = body.clone();
        vector::normalize_write(&mut normalized, &self.schema)?;
        let object = normalized
            .as_object()
            .ok_or("write body must be an object")?;
        if let Some(metric) = object.get("distance_metric") {
            let metric = metric.as_str().ok_or("distance_metric must be a string")?;
            if !matches!(metric, "cosine_distance" | "euclidean_squared") {
                return Err(WriteError::Invalid(
                    "distance_metric must be cosine_distance or euclidean_squared".into(),
                ));
            }
            if !self
                .schema
                .values()
                .any(|v| vector::dimensions(v).is_some() || has_embed(v))
            {
                return Err(WriteError::Invalid(
                    "distance_metric requires a vector attribute".into(),
                ));
            }
            self.distance_metric = Some(metric.into());
        }
        for (condition, operations) in [
            ("upsert_condition", &["upsert_rows", "upsert_columns"][..]),
            (
                "patch_condition",
                &["patch_rows", "patch_columns", "patch_by_filter"][..],
            ),
            ("delete_condition", &["deletes", "delete_by_filter"][..]),
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
                if filter_matches(filter, row)
                    && condition_matches(object, "delete_condition", Some(row))
                {
                    deleted_ids.push(row["id"].clone());
                    false
                } else {
                    true
                }
            });
        }
        if let Some(deletes) = object.get("deletes") {
            let deletes = deletes.as_array().ok_or("deletes must be an array")?;
            for id in deletes {
                let key = id_key(id)?;
                if condition_matches(object, "delete_condition", self.rows.get(&key))
                    && self.rows.remove(&key).is_some()
                {
                    deleted_ids.push(id.clone());
                }
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
                if filter_matches(filter, row)
                    && condition_matches(object, "patch_condition", Some(row))
                {
                    for (field, value) in values {
                        row.insert(field.clone(), value.clone());
                    }
                    patched_ids.push(row["id"].clone());
                }
            }
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
                if !condition_matches(object, "patch_condition", self.rows.get(&key)) {
                    continue;
                }
                if let Some(row) = self.rows.get_mut(&key) {
                    for (field, value) in patch {
                        if field != "id" {
                            row.insert(field.clone(), value.clone());
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
                if !condition_matches(object, "upsert_condition", self.rows.get(&id_key(&id)?)) {
                    continue;
                }
                for (field, value) in &row {
                    if !self.schema.contains_key(field) {
                        self.schema.insert(field.clone(), infer_type(value)?);
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
            validate_value(field, value, definition)?;
        }
        Ok(())
    }
}

fn condition_matches(
    object: &Map<String, Value>,
    condition: &str,
    current: Option<&Map<String, Value>>,
) -> bool {
    object
        .get(condition)
        .is_none_or(|filter| filter_matches(filter, current.unwrap_or(&Map::new())))
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

fn validate_definition(field: &str, definition: &Value) -> Result<(), String> {
    if let Some(config) = definition.as_object() {
        for (option, value) in config {
            match option.as_str() {
                "type" if !value.is_string() => {
                    return Err(format!(
                        "schema option type must be a string for attribute {field}"
                    ));
                }
                "filterable" | "ann" | "regex" | "glob" | "full_text_search"
                    if !value.is_boolean() =>
                {
                    return Err(format!(
                        "schema option {option} must be a boolean for attribute {field}"
                    ));
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
                "type" | "filterable" | "ann" | "regex" | "glob" | "full_text_search" => {}
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
    match field_type(definition) {
        "uuid" | "uint" | "int" | "float" | "string" | "bool" | "datetime" | "[]uuid"
        | "[]uint" | "[]int" | "[]float" | "[]string" | "[]bool" | "[]datetime" => Ok(()),
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

fn validate_id(value: &Value, definition: Option<&Value>) -> Result<(), String> {
    id_key(value)?;
    if let Some(definition) = definition {
        validate_value("id", value, definition)?;
    }
    Ok(())
}

fn id_key(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) if !s.is_empty() => Ok(format!("s:{s}")),
        Value::Number(n) if n.as_u64().is_some() => Ok(format!("n:{n}")),
        _ => Err("id must be a string or unsigned integer".into()),
    }
}

fn infer_type(value: &Value) -> Result<Value, String> {
    let name = match value {
        Value::String(_) => "string",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.as_u64().is_some() => "uint",
        Value::Number(_) => "float",
        Value::Null => return Err("cannot infer schema from null attribute".into()),
        _ => return Err("cannot infer schema from array or object attribute".into()),
    };
    Ok(json!(name))
}

fn validate_value(field: &str, value: &Value, definition: &Value) -> Result<(), String> {
    if value.is_null() && field != "id" {
        return Ok(());
    }
    let valid = match field_type(definition) {
        "uuid" => value.as_str().is_some_and(uuid_like),
        "uint" => value.as_u64().is_some(),
        "int" => value.as_i64().is_some(),
        "float" => value.is_number(),
        "string" | "datetime" => value.is_string(),
        "bool" => value.is_boolean(),
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
                            item.as_f64()
                                .is_some_and(|number| (-128.0..=127.0).contains(&number))
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
