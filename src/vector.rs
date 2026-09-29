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
        if let Some(value) = row.get(field) {
            let value = if let (Some(dimensions), Some(encoded)) =
                (dimensions(definition), value.as_str())
            {
                decode(encoded, dimensions)?
            } else {
                value.clone()
            };
            if let Some(normalized) = normalize_array_value(&value, definition)? {
                row.insert(field.clone(), normalized);
            }
        }
    }
    Ok(())
}

pub(crate) fn normalize_write(
    body: &mut Value,
    schema: &Map<String, Value>,
    generated_vectors: &Map<String, Value>,
) -> Result<(), String> {
    let object = body.as_object_mut().ok_or("write body must be an object")?;
    for key in ["upsert_rows", "patch_rows"] {
        if let Some(rows) = object.get_mut(key).and_then(Value::as_array_mut) {
            for row in rows {
                if let Some(row) = row.as_object_mut() {
                    normalize_row(row, schema)?;
                    if key == "upsert_rows" {
                        normalize_row(row, generated_vectors)?;
                    }
                }
            }
        }
    }
    for key in ["upsert_columns", "patch_columns"] {
        if let Some(columns) = object.get_mut(key).and_then(Value::as_object_mut) {
            for (field, definition) in schema.iter().chain(
                (key == "upsert_columns")
                    .then_some(generated_vectors.iter())
                    .into_iter()
                    .flatten(),
            ) {
                if let Some(values) = columns.get_mut(field).and_then(Value::as_array_mut) {
                    for value in values {
                        if let (Some(dimensions), Some(encoded)) =
                            (dimensions(definition), value.as_str())
                        {
                            *value = decode(encoded, dimensions)?;
                        }
                        if let Some(normalized) = normalize_array_value(value, definition)? {
                            *value = normalized;
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

fn normalize_array_value(value: &Value, definition: &Value) -> Result<Option<Value>, String> {
    let kind = definition
        .as_str()
        .or_else(|| definition.get("type").and_then(Value::as_str))
        .unwrap_or("");
    let convert = |element: &Value| -> Result<Value, String> {
        let number = element.as_f64().ok_or("vector must contain numbers")? as f32;
        if !number.is_finite() {
            return Err("vector must contain finite floats".into());
        }
        if kind.ends_with("]i8") {
            if number.fract() != 0.0 || !(-128.0..=127.0).contains(&number) {
                return Err("i8 vector element is not an integer in range".into());
            }
            Ok(json!(number as i8))
        } else if kind.ends_with("]f16") {
            let number = f16::from_f32(number).to_f32();
            if !number.is_finite() {
                return Err("f16 vector element is out of range".into());
            }
            Ok(float_value(number))
        } else {
            Ok(float_value(number))
        }
    };
    if kind == "{}f16" {
        return value
            .as_object()
            .map(|weights| {
                weights
                    .iter()
                    .map(|(token, weight)| {
                        let number = weight
                            .as_f64()
                            .ok_or("sparse vector weights must be numbers")?
                            as f32;
                        let rounded = f16::from_f32(number).to_f32();
                        if !rounded.is_finite() {
                            return Err("sparse vector weight is out of range".into());
                        }
                        Ok((token.clone(), float_value(rounded)))
                    })
                    .collect::<Result<Map<_, _>, String>>()
                    .map(|weights| Some(Value::Object(weights)))
            })
            .transpose()
            .map(Option::flatten);
    }
    if dimensions(definition).is_some() {
        return value
            .as_array()
            .map(|elements| {
                elements
                    .iter()
                    .map(convert)
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| Some(Value::Array(values)))
            })
            .transpose()
            .map(Option::flatten);
    }
    if multi_dimensions(definition).is_some() {
        return value
            .as_array()
            .map(|vectors| {
                vectors
                    .iter()
                    .map(|vector| {
                        vector
                            .as_array()
                            .ok_or("multi-vector must contain vector arrays")?
                            .iter()
                            .map(&convert)
                            .collect::<Result<Vec<_>, _>>()
                            .map(Value::Array)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|values| Some(Value::Array(values)))
            })
            .transpose()
            .map(Option::flatten);
    }
    Ok(None)
}

/// JSON numbers are stored as f64. Parse the shortest f32 round-trip decimal so
/// float vector responses have the same precision as the live f32/f16 output.
fn float_value(number: f32) -> Value {
    json!(serialized_f32(number))
}

pub(crate) fn serialized_f32(number: f32) -> f64 {
    number.to_string().parse().unwrap()
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
            if parts.len() == 3 && parts[1] == "SparseKNN" {
                if let Some(definition) = parts[0].as_str().and_then(|field| schema.get(field)) {
                    if let Some(normalized) = normalize_array_value(&parts[2], definition)? {
                        parts[2] = normalized;
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
