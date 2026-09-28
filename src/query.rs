use crate::store::{field_type, known_field, Namespace};
use chrono::{DateTime, FixedOffset};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

impl Namespace {
    pub fn query(&self, body: &Value) -> Result<Value, String> {
        let object = body.as_object().ok_or("query body must be an object")?;
        if let Some(queries) = object.get("queries") {
            if object.len() != 1 {
                return Err("queries cannot be combined with other query fields".into());
            }
            let queries = queries.as_array().ok_or("queries must be an array")?;
            for query in queries {
                self.validate_query(query)?;
            }
            let results = queries
                .iter()
                .map(|q| self.execute_query(q))
                .collect::<Vec<_>>();
            Ok(json!({"results": results}))
        } else {
            self.validate_query(body)?;
            Ok(self.execute_query(body))
        }
    }

    fn validate_query(&self, body: &Value) -> Result<(), String> {
        let object = body.as_object().ok_or("subquery must be an object")?;
        let rank = object.get("rank_by").ok_or("rank_by is required")?;
        validate_rank(rank, &self.schema)?;
        if let Some(filter) = object.get("filters") {
            validate_filter(filter, &self.schema)?;
        }
        if let Some(include) = object.get("include_attributes") {
            if !include.is_boolean() {
                let fields = include
                    .as_array()
                    .ok_or("include_attributes must be an array or boolean")?;
                for field in fields {
                    let name = field
                        .as_str()
                        .ok_or("include_attributes entries must be strings")?;
                    if !known_field(&self.schema, name) {
                        return Err(format!("attribute {name} does not exist in schema"));
                    }
                }
            }
        }
        match object.get("limit") {
            Some(Value::Number(n)) if n.as_u64().is_some_and(|v| v <= 10_000) => (),
            _ => return Err("limit must be an integer at most 10000".into()),
        }
        Ok(())
    }

    fn execute_query(&self, body: &Value) -> Value {
        let object = body.as_object().unwrap();
        let rank = &object["rank_by"];
        let filter = object.get("filters");
        let ascending = is_ann(rank) || is_ascending(rank);
        let mut scored = self
            .rows
            .values()
            .filter_map(|row| {
                if filter.is_some_and(|f| !filter_matches(f, row)) {
                    return None;
                }
                let score = score_rank(rank, row, &self.rows);
                if !score.is_finite() || (!is_ascending(rank) && score <= 0.0 && !is_ann(rank)) {
                    return None;
                }
                Some((score, row))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|(a, row_a), (b, row_b)| {
            let order = a.total_cmp(b);
            (if ascending { order } else { order.reverse() })
                .then_with(|| row_a["id"].to_string().cmp(&row_b["id"].to_string()))
        });
        let offset = object.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = object["limit"].as_u64().unwrap() as usize;
        let rows = scored
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(score, row)| {
                let mut result = Map::new();
                result.insert("id".into(), row["id"].clone());
                match object.get("include_attributes") {
                    Some(Value::Bool(true)) => {
                        for (key, value) in row {
                            result.insert(key.clone(), value.clone());
                        }
                    }
                    Some(Value::Array(fields)) => {
                        for field in fields {
                            let field = field.as_str().unwrap();
                            if let Some(value) = row.get(field) {
                                result.insert(field.into(), value.clone());
                            }
                        }
                    }
                    _ => (),
                }
                if !is_ascending(rank) {
                    result.insert("$dist".into(), json!(score));
                }
                Value::Object(result)
            })
            .collect::<Vec<_>>();
        json!({"rows": rows})
    }
}

pub(crate) fn validate_filter(filter: &Value, schema: &Map<String, Value>) -> Result<(), String> {
    let parts = filter.as_array().ok_or("filter must be an array")?;
    if parts.len() == 2 && parts[0] == "And" {
        let children = parts[1].as_array().ok_or("And requires an array")?;
        for child in children {
            validate_filter(child, schema)?;
        }
        return Ok(());
    }
    if parts.len() != 3 {
        return Err("filter must have three elements".into());
    }
    let field = parts[0]
        .as_str()
        .ok_or("filter attribute must be a string")?;
    if !known_field(schema, field) {
        return Err(format!("attribute {field} does not exist in schema"));
    }
    let op = parts[1]
        .as_str()
        .ok_or("filter operator must be a string")?;
    match op {
        "Eq" | "NotEq" | "Gte" | "Lte" => Ok(()),
        "In" if parts[2].is_array() => Ok(()),
        "In" => Err("In requires an array".into()),
        _ => Err(format!("unsupported filter operator {op}")),
    }
}

pub(crate) fn filter_matches(filter: &Value, row: &Map<String, Value>) -> bool {
    let parts = filter.as_array().unwrap();
    if parts.len() == 2 {
        return parts[1]
            .as_array()
            .unwrap()
            .iter()
            .all(|f| filter_matches(f, row));
    }
    let field = parts[0].as_str().unwrap();
    let left = row.get(field).unwrap_or(&Value::Null);
    let right = &parts[2];
    match parts[1].as_str().unwrap() {
        "Eq" => left == right,
        "NotEq" => left != right,
        "In" => right.as_array().unwrap().contains(left),
        "Gte" => !left.is_null() && compare_values(left, right).is_some_and(|o| o >= 0),
        "Lte" => left.is_null() || compare_values(left, right).is_some_and(|o| o <= 0),
        _ => false,
    }
}

fn compare_values(left: &Value, right: &Value) -> Option<i8> {
    if let (Some(a), Some(b)) = (left.as_f64(), right.as_f64()) {
        return Some(if a < b {
            -1
        } else if a > b {
            1
        } else {
            0
        });
    }
    let (a, b) = (left.as_str()?, right.as_str()?);
    if let (Ok(a), Ok(b)) = (
        DateTime::<FixedOffset>::parse_from_rfc3339(a),
        DateTime::<FixedOffset>::parse_from_rfc3339(b),
    ) {
        return Some(if a < b {
            -1
        } else if a > b {
            1
        } else {
            0
        });
    }
    Some(if a < b {
        -1
    } else if a > b {
        1
    } else {
        0
    })
}

fn validate_rank(rank: &Value, schema: &Map<String, Value>) -> Result<(), String> {
    let parts = rank.as_array().ok_or("rank_by must be an array")?;
    if parts.len() == 2 && parts[0] == "Sum" {
        let children = parts[1].as_array().ok_or("Sum requires an array")?;
        for child in children {
            validate_rank(child, schema)?;
        }
        return Ok(());
    }
    if parts.len() == 3 && parts[0] == "Product" {
        if !parts[1].is_number() {
            return Err("Product requires a numeric weight".into());
        }
        return validate_rank(&parts[2], schema);
    }
    if parts.len() != 3 && parts.len() != 2 {
        return Err("unsupported rank_by expression".into());
    }
    let field = parts[0].as_str().ok_or("rank attribute must be a string")?;
    if !known_field(schema, field) {
        return Err(format!("attribute {field} does not exist in schema"));
    }
    let operator = parts[1].as_str().ok_or("rank operator must be a string")?;
    match operator {
        "ANN"
            if parts.len() == 3
                && parts[2]
                    .as_array()
                    .is_some_and(|a| a.iter().all(Value::is_number)) =>
        {
            if !(field.starts_with("embed_") || field_type(&schema[field]).starts_with('[')) {
                return Err(format!("attribute {field} is not a vector"));
            }
            let dimensions = field.strip_prefix("embed_").map_or_else(
                || {
                    field_type(&schema[field])
                        .trim_start_matches('[')
                        .split(']')
                        .next()
                        .and_then(|n| n.parse::<usize>().ok())
                },
                |base| {
                    schema[base]
                        .get("embed")
                        .and_then(|v| v.get("dims"))
                        .and_then(Value::as_u64)
                        .map(|n| n as usize)
                },
            );
            if dimensions != parts[2].as_array().map(Vec::len) {
                return Err(format!(
                    "query vector has wrong dimensions for attribute {field}"
                ));
            }
            Ok(())
        }
        "BM25" if parts.len() == 3 && parts[2].is_string() => {
            let definition = &schema[field];
            if field_type(definition) != "string"
                || definition.get("full_text_search") != Some(&Value::Bool(true))
            {
                return Err(format!("attribute {field} has no full-text index"));
            }
            Ok(())
        }
        "asc" | "desc" if parts.len() == 2 => Ok(()),
        _ => Err(format!("unsupported rank operator {operator}")),
    }
}

fn is_ann(rank: &Value) -> bool {
    rank.as_array()
        .is_some_and(|a| a.len() == 3 && a[1] == "ANN")
}
fn is_ascending(rank: &Value) -> bool {
    rank.as_array()
        .is_some_and(|a| a.len() == 2 && a[1] == "asc")
}

fn score_rank(
    rank: &Value,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
) -> f64 {
    let parts = rank.as_array().unwrap();
    if parts[0] == "Sum" {
        return parts[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|child| score_rank(child, row, corpus))
            .sum();
    }
    if parts[0] == "Product" {
        return parts[1].as_f64().unwrap() * score_rank(&parts[2], row, corpus);
    }
    let field = parts[0].as_str().unwrap();
    match parts[1].as_str().unwrap() {
        "ANN" => {
            let query = parts[2].as_array().unwrap();
            let vector = row.get(field).and_then(Value::as_array);
            match vector {
                Some(vector) if vector.len() == query.len() => {
                    let dot: f64 = vector
                        .iter()
                        .zip(query)
                        .map(|(a, b)| a.as_f64().unwrap() * b.as_f64().unwrap())
                        .sum();
                    let norm_a: f64 = vector
                        .iter()
                        .map(|a| a.as_f64().unwrap().powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let norm_b: f64 = query
                        .iter()
                        .map(|a| a.as_f64().unwrap().powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if norm_a == 0.0 || norm_b == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 - dot / norm_a / norm_b
                    }
                }
                _ => f64::INFINITY,
            }
        }
        "BM25" => bm25(field, parts[2].as_str().unwrap(), row, corpus),
        "asc" | "desc" => row.get(field).and_then(Value::as_f64).unwrap_or(0.0),
        _ => 0.0,
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn bm25(
    field: &str,
    query: &str,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
) -> f64 {
    let Some(text) = row.get(field).and_then(Value::as_str) else {
        return 0.0;
    };
    let doc_tokens = tokens(text);
    let query_tokens = tokens(query);
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let lengths = corpus
        .values()
        .filter_map(|r| {
            r.get(field)
                .and_then(Value::as_str)
                .map(|s| tokens(s).len())
        })
        .collect::<Vec<_>>();
    let avg_len = (lengths.iter().sum::<usize>() as f64 / lengths.len().max(1) as f64).max(1.0);
    let mut terms = HashMap::new();
    for token in query_tokens {
        *terms.entry(token).or_insert(0_usize) += 1;
    }
    terms
        .into_iter()
        .map(|(term, query_freq)| {
            let freq = doc_tokens.iter().filter(|token| **token == term).count() as f64;
            if freq == 0.0 {
                return 0.0;
            }
            let df = corpus
                .values()
                .filter(|r| {
                    r.get(field)
                        .and_then(Value::as_str)
                        .is_some_and(|s| tokens(s).contains(&term))
                })
                .count() as f64;
            let n = corpus.len() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let norm = 1.2 * (1.0 - 0.75 + 0.75 * doc_tokens.len() as f64 / avg_len);
            query_freq as f64 * idf * freq * 2.2 / (freq + norm)
        })
        .sum()
}
