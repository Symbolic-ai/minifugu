use base64::{engine::general_purpose::STANDARD, Engine};
use half::f16;
use serde_json::{json, Map, Value};

pub(crate) fn dimensions(definition: &Value) -> Option<usize> {
    let kind = definition
        .as_str()
        .or_else(|| definition.get("type")?.as_str())?;
    if !(kind.ends_with("]f16") || kind.ends_with("]f32") || kind.ends_with("]i8")) {
        return None;
    }
    kind.strip_prefix('[')?.split(']').next()?.parse().ok()
}

pub(crate) fn multi_dimensions(definition: &Value) -> Option<usize> {
    let kind = definition
        .as_str()
        .or_else(|| definition.get("type")?.as_str())?;
    kind.strip_prefix("[][")?.strip_suffix("]f32")?.parse().ok()
}

/// Turbopuffer's documented maximum attribute value size. It bounds an array by its total
/// byte size; the service sets no limit on the element count.
const MAX_ATTRIBUTE_BYTES: usize = 8 * 1024 * 1024;

/// Whether `tokens` float32 vectors of `dimensions` fit in one attribute value.
pub(crate) fn multi_vector_within_limit(tokens: usize, dimensions: usize) -> bool {
    tokens
        .checked_mul(dimensions)
        .and_then(|elements| elements.checked_mul(4))
        .is_some_and(|bytes| bytes <= MAX_ATTRIBUTE_BYTES)
}

pub(crate) fn decode(value: &str, dimensions: usize) -> Result<Value, String> {
    let bytes = STANDARD
        .decode(value)
        .map_err(|_| "invalid base64 vector")?;
    if bytes.len() != dimensions * 4 {
        return Err("base64 vector has wrong dimensions".into());
    }
    let floats = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    if floats.iter().any(|value| !value.is_finite()) {
        return Err("base64 vector must contain finite floats".into());
    }
    Ok(json!(floats))
}

pub(crate) fn encode(value: &Value, definition: &Value) -> Result<Value, String> {
    let values = value.as_array().ok_or("vector must be an array")?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    let kind = definition
        .as_str()
        .or_else(|| definition.get("type").and_then(Value::as_str))
        .ok_or("vector type is missing")?;
    for value in values {
        let number = value.as_f64().ok_or("vector must contain numbers")? as f32;
        if !number.is_finite() {
            return Err("vector must contain finite numbers".into());
        }
        if kind.ends_with("]i8") {
            if number.fract() != 0.0 || !(-128.0..=127.0).contains(&number) {
                return Err("i8 vector element is not an integer in range".into());
            }
            bytes.push(number as i8 as u8);
        } else if kind.ends_with("]f16") {
            bytes.extend_from_slice(&f16::from_f32(number).to_bits().to_le_bytes());
        } else {
            bytes.extend_from_slice(&number.to_le_bytes());
        }
    }
    Ok(Value::String(STANDARD.encode(bytes)))
}

pub(crate) fn normalize_row(
    row: &mut Map<String, Value>,
    schema: &Map<String, Value>,
) -> Result<(), String> {
    for (field, definition) in schema {
        if let (Some(dimensions), Some(Value::String(encoded))) =
            (dimensions(definition), row.get(field))
        {
            row.insert(field.clone(), decode(encoded, dimensions)?);
        }
    }
    Ok(())
}

pub(crate) fn normalize_write(body: &mut Value, schema: &Map<String, Value>) -> Result<(), String> {
    let object = body.as_object_mut().ok_or("write body must be an object")?;
    for key in ["upsert_rows", "patch_rows"] {
        if let Some(rows) = object.get_mut(key).and_then(Value::as_array_mut) {
            for row in rows {
                if let Some(row) = row.as_object_mut() {
                    normalize_row(row, schema)?;
                }
            }
        }
    }
    for key in ["upsert_columns", "patch_columns"] {
        if let Some(columns) = object.get_mut(key).and_then(Value::as_object_mut) {
            for (field, definition) in schema {
                if let (Some(dimensions), Some(values)) = (
                    dimensions(definition),
                    columns.get_mut(field).and_then(Value::as_array_mut),
                ) {
                    for value in values {
                        if let Some(encoded) = value.as_str() {
                            *value = decode(encoded, dimensions)?;
                        }
                    }
                }
            }
        }
    }
    if let Some(patch) = object
        .get_mut("patch_by_filter")
        .and_then(|value| value.get_mut("patch"))
        .and_then(Value::as_object_mut)
    {
        normalize_row(patch, schema)?;
    }
    Ok(())
}

pub(crate) fn normalize_query(
    value: &mut Value,
    schema: &Map<String, Value>,
) -> Result<(), String> {
    match value {
        Value::Array(parts) => {
            if parts.len() == 3 && matches!(parts[1].as_str(), Some("ANN" | "kNN" | "VectorDist")) {
                if let (Some(field), Some(encoded)) = (parts[0].as_str(), parts[2].as_str()) {
                    if let Some(dimensions) = schema.get(field).and_then(dimensions) {
                        parts[2] = decode(encoded, dimensions)?;
                    }
                }
            }
            for part in parts {
                normalize_query(part, schema)?;
            }
        }
        Value::Object(object) => {
            for value in object.values_mut() {
                normalize_query(value, schema)?;
            }
        }
        _ => (),
    }
    Ok(())
}

pub(crate) fn encode_response(
    value: &mut Value,
    schema: &Map<String, Value>,
) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            for (field, item) in object {
                if let Some(definition) = schema
                    .get(field)
                    .filter(|definition| dimensions(definition).is_some() && item.is_array())
                {
                    *item = encode(item, definition)?;
                } else {
                    encode_response(item, schema)?;
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                encode_response(item, schema)?;
            }
        }
        _ => (),
    }
    Ok(())
}
