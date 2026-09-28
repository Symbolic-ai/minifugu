use crate::embedding::EmbeddingMode;
use crate::query::{filter_matches, validate_filter};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

#[derive(Clone, Default)]
pub struct Namespace {
    pub(crate) schema: Map<String, Value>,
    pub(crate) rows: BTreeMap<String, Map<String, Value>>,
    pub(crate) distance_metric: Option<String>,
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
        if let Some(metric) = object.get("distance_metric") {
            let metric = metric.as_str().ok_or("distance_metric must be a string")?;
            if metric != "cosine_distance" {
                return Err(WriteError::Invalid(
                    "only cosine_distance is supported".into(),
                ));
            }
            if !self
                .schema
                .values()
                .any(|v| field_type(v).starts_with('[') || has_embed(v))
            {
                return Err(WriteError::Invalid(
                    "distance_metric requires a vector attribute".into(),
                ));
            }
            self.distance_metric = Some(metric.into());
        }
        let mut affected = 0;
        if let Some(filter) = object.get("delete_by_filter") {
            validate_filter(filter, &self.schema)?;
            let old_count = self.rows.len();
            self.rows.retain(|_, row| !filter_matches(filter, row));
            affected += old_count - self.rows.len();
        }
        if let Some(deletes) = object.get("deletes") {
            let deletes = deletes.as_array().ok_or("deletes must be an array")?;
            for id in deletes {
                if self.rows.remove(&id_key(id)?).is_some() {
                    affected += 1;
                }
            }
        }
        if let Some(rows) = object.get("upsert_rows") {
            let rows = rows.as_array().ok_or("upsert_rows must be an array")?;
            for row in rows {
                let mut row = row
                    .as_object()
                    .ok_or("each upsert row must be an object")?
                    .clone();
                let id = row.get("id").ok_or("upsert row requires id")?.clone();
                validate_id(&id, self.schema.get("id"))?;
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
                affected += 1;
            }
        }
        Ok(json!({"rows_affected": affected}))
    }
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
        "uuid" | "uint" | "int" | "float" | "string" | "bool" | "datetime" | "[1536]f16" => Ok(()),
        value if value.starts_with('[') && value.ends_with("]f16") => Ok(()),
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
        vector if vector.starts_with('[') => {
            let expected = vector
                .trim_start_matches('[')
                .split(']')
                .next()
                .and_then(|n| n.parse::<usize>().ok());
            value.as_array().is_some_and(|array| {
                Some(array.len()) == expected && array.iter().all(Value::is_number)
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
