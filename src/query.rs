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
        if object.contains_key("aggregate_by") {
            return self.validate_aggregation(object);
        }
        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "rank_by"
                    | "filters"
                    | "include_attributes"
                    | "exclude_attributes"
                    | "limit"
                    | "top_k"
                    | "offset"
            ) {
                return Err(format!("unsupported query field {key}"));
            }
        }
        if let Some(rank) = object.get("rank_by") {
            validate_rank(rank, &self.schema)?;
        }
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
        if let Some(exclude) = object.get("exclude_attributes") {
            let fields = exclude
                .as_array()
                .ok_or("exclude_attributes must be an array")?;
            if fields.iter().any(|field| !field.is_string()) {
                return Err("exclude_attributes entries must be strings".into());
            }
        }
        if object.contains_key("include_attributes") && object.contains_key("exclude_attributes") {
            return Err("include_attributes and exclude_attributes cannot be combined".into());
        }
        if object.contains_key("limit") && object.contains_key("top_k") {
            return Err("limit and top_k cannot be combined".into());
        }
        let limit = object.get("limit").or_else(|| object.get("top_k"));
        match limit {
            Some(Value::Number(n)) if n.as_u64().is_some_and(|v| v <= 10_000) => (),
            Some(Value::Object(v))
                if v.get("total")
                    .and_then(Value::as_u64)
                    .is_some_and(|n| n <= 10_000) =>
            {
                if let Some(per) = v.get("per") {
                    let per = per.as_object().ok_or("limit.per must be an object")?;
                    let fields = per
                        .get("attributes")
                        .and_then(Value::as_array)
                        .ok_or("limit.per requires attributes")?;
                    if fields.iter().any(|field| {
                        !field
                            .as_str()
                            .is_some_and(|name| known_field(&self.schema, name))
                    }) {
                        return Err("limit.per contains an unknown attribute".into());
                    }
                    for field in fields {
                        let field = field.as_str().unwrap();
                        let included = match object.get("include_attributes") {
                            Some(Value::Bool(true)) => true,
                            Some(Value::Array(selected)) => selected.contains(&json!(field)),
                            _ => object
                                .get("exclude_attributes")
                                .and_then(Value::as_array)
                                .is_some_and(|excluded| !excluded.contains(&json!(field))),
                        };
                        if field != "id" && !included {
                            return Err(format!(
                                "limit.per attribute {field} must be included in response"
                            ));
                        }
                    }
                    if !per
                        .get("limit")
                        .and_then(Value::as_u64)
                        .is_some_and(|n| n > 0 && n <= 10_000)
                    {
                        return Err("limit.per requires a positive limit".into());
                    }
                }
            }
            _ => return Err("limit or top_k must be an integer at most 10000".into()),
        }
        if object.get("offset").is_some_and(|v| v.as_u64().is_none()) {
            return Err("offset must be a nonnegative integer".into());
        }
        Ok(())
    }

    fn execute_query(&self, body: &Value) -> Value {
        let object = body.as_object().unwrap();
        if object.contains_key("aggregate_by") {
            return self.execute_aggregation(object);
        }
        let rank = object.get("rank_by");
        let filter = object.get("filters");
        let ascending = rank.is_none_or(|rank| is_ann(rank) || is_ascending(rank));
        let mut scored = self
            .rows
            .values()
            .filter_map(|row| {
                if filter.is_some_and(|f| !filter_matches(f, row)) {
                    return None;
                }
                let score = rank.map_or(0.0, |rank| score_rank(rank, row, &self.rows));
                if !score.is_finite()
                    || rank.is_some_and(|rank| {
                        !is_ascending(rank)
                            && score <= 0.0
                            && !is_ann(rank)
                            && !is_attribute_order(rank)
                    })
                {
                    return None;
                }
                Some((score, row))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|(a, row_a), (b, row_b)| {
            if let Some(rank) = rank.filter(|rank| is_attribute_order(rank)) {
                return compare_attribute_order(rank, row_a, row_b)
                    .then_with(|| row_a["id"].to_string().cmp(&row_b["id"].to_string()));
            }
            let order = a.total_cmp(b);
            (if ascending { order } else { order.reverse() })
                .then_with(|| row_a["id"].to_string().cmp(&row_b["id"].to_string()))
        });
        let offset = object.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = object
            .get("limit")
            .or_else(|| object.get("top_k"))
            .and_then(|v| {
                v.as_u64()
                    .or_else(|| v.get("total").and_then(Value::as_u64))
            })
            .unwrap() as usize;
        let per = object.get("limit").and_then(|v| v.get("per"));
        let mut per_counts = HashMap::<String, usize>::new();
        let rows = scored
            .into_iter()
            .filter(|(_, row)| {
                let Some(per) = per else {
                    return true;
                };
                let key = per["attributes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|field| {
                        row.get(field.as_str().unwrap())
                            .cloned()
                            .unwrap_or(Value::Null)
                    })
                    .collect::<Vec<_>>();
                let key = serde_json::to_string(&key).unwrap();
                let count = per_counts.entry(key).or_default();
                if *count >= per["limit"].as_u64().unwrap() as usize {
                    false
                } else {
                    *count += 1;
                    true
                }
            })
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
                if let Some(Value::Array(fields)) = object.get("exclude_attributes") {
                    for (field, value) in row {
                        if field != "id" && !fields.contains(&Value::String(field.clone())) {
                            result.insert(field.clone(), value.clone());
                        }
                    }
                }
                if rank.is_some_and(|rank| !is_attribute_order(rank) && !is_ascending(rank)) {
                    result.insert("$dist".into(), json!(score));
                }
                Value::Object(result)
            })
            .collect::<Vec<_>>();
        json!({"rows": rows})
    }

    fn validate_aggregation(&self, object: &Map<String, Value>) -> Result<(), String> {
        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "aggregate_by" | "group_by" | "filters" | "top_k"
            ) {
                return Err(format!("unsupported aggregation field {key}"));
            }
        }
        let aggregates = object["aggregate_by"]
            .as_object()
            .ok_or("aggregate_by must be an object")?;
        if aggregates.is_empty() || aggregates.len() > 8 {
            return Err("aggregate_by requires 1 to 8 functions".into());
        }
        for (name, expression) in aggregates {
            let parts = expression
                .as_array()
                .ok_or_else(|| format!("aggregate {name} must be an array"))?;
            match parts.as_slice() {
                [operator] if operator == "Count" => (),
                [operator, field] if operator == "Sum" => {
                    let field = field.as_str().ok_or("Sum attribute must be a string")?;
                    let definition = self
                        .schema
                        .get(field)
                        .ok_or_else(|| format!("attribute {field} does not exist in schema"))?;
                    if !matches!(field_type(definition), "int" | "uint" | "float") {
                        return Err(format!("Sum attribute {field} must be numeric"));
                    }
                }
                _ => return Err(format!("unsupported aggregate {name}")),
            }
        }
        if let Some(filter) = object.get("filters") {
            validate_filter(filter, &self.schema)?;
        }
        if let Some(groups) = object.get("group_by") {
            let groups = groups.as_array().ok_or("group_by must be an array")?;
            for field in groups {
                let field = field
                    .as_str()
                    .ok_or("group_by entries must be attribute names")?;
                if !known_field(&self.schema, field) {
                    return Err(format!("attribute {field} does not exist in schema"));
                }
            }
        }
        if object
            .get("top_k")
            .is_some_and(|value| !value.as_u64().is_some_and(|n| n <= 10_000))
        {
            return Err("top_k must be an integer at most 10000".into());
        }
        Ok(())
    }

    fn execute_aggregation(&self, object: &Map<String, Value>) -> Value {
        let rows = self
            .rows
            .values()
            .filter(|row| {
                object
                    .get("filters")
                    .is_none_or(|filter| filter_matches(filter, row))
            })
            .collect::<Vec<_>>();
        let aggregates = object["aggregate_by"].as_object().unwrap();
        let Some(groups) = object.get("group_by") else {
            return json!({"aggregations": aggregate_values(aggregates, &rows), "billing":{}, "performance":{"server_total_ms":0}});
        };
        let fields = groups.as_array().unwrap();
        let mut grouped = std::collections::BTreeMap::<
            String,
            (Map<String, Value>, Vec<&Map<String, Value>>),
        >::new();
        for row in rows {
            let values = fields
                .iter()
                .map(|field| {
                    row.get(field.as_str().unwrap())
                        .cloned()
                        .unwrap_or(Value::Null)
                })
                .collect::<Vec<_>>();
            let key = serde_json::to_string(&values).unwrap();
            let entry = grouped.entry(key).or_insert_with(|| {
                let attrs = fields
                    .iter()
                    .zip(&values)
                    .map(|(field, value)| (field.as_str().unwrap().to_owned(), value.clone()))
                    .collect();
                (attrs, Vec::new())
            });
            entry.1.push(row);
        }
        let limit = object
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(10_000) as usize;
        let groups = grouped
            .into_values()
            .take(limit)
            .map(|(mut attrs, rows)| {
                attrs.extend(aggregate_values(aggregates, &rows));
                Value::Object(attrs)
            })
            .collect::<Vec<_>>();
        json!({"aggregation_groups":groups, "billing":{}, "performance":{"server_total_ms":0}})
    }
}

fn aggregate_values(
    aggregates: &Map<String, Value>,
    rows: &[&Map<String, Value>],
) -> Map<String, Value> {
    aggregates
        .iter()
        .map(|(name, expression)| {
            let parts = expression.as_array().unwrap();
            let value = if parts[0] == "Count" {
                json!(rows.len())
            } else {
                let field = parts[1].as_str().unwrap();
                let total: f64 = rows
                    .iter()
                    .filter_map(|row| row.get(field).and_then(Value::as_f64))
                    .sum();
                if total.fract() == 0.0 && total >= i64::MIN as f64 && total <= i64::MAX as f64 {
                    json!(total as i64)
                } else {
                    json!(total)
                }
            };
            (name.clone(), value)
        })
        .collect()
}

pub(crate) fn validate_filter(filter: &Value, schema: &Map<String, Value>) -> Result<(), String> {
    let parts = filter.as_array().ok_or("filter must be an array")?;
    if parts.len() == 2 && (parts[0] == "And" || parts[0] == "Or") {
        let children = parts[1].as_array().ok_or("And/Or requires an array")?;
        for child in children {
            validate_filter(child, schema)?;
        }
        return Ok(());
    }
    if parts.len() == 2 && parts[0] == "Not" {
        return validate_filter(&parts[1], schema);
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
    if matches!(
        op,
        "Contains" | "NotContains" | "ContainsAny" | "NotContainsAny"
    ) && !schema
        .get(field)
        .is_some_and(|definition| field_type(definition).starts_with("[]"))
    {
        return Err(format!("attribute {field} is not an array"));
    }
    match op {
        "Eq" | "NotEq" | "Gt" | "Gte" | "Lt" | "Lte" | "Contains" | "NotContains" => Ok(()),
        "In" | "ContainsAny" | "NotContainsAny" if parts[2].is_array() => Ok(()),
        "In" | "ContainsAny" | "NotContainsAny" => Err(format!("{op} requires an array")),
        _ => Err(format!("unsupported filter operator {op}")),
    }
}

pub(crate) fn filter_matches(filter: &Value, row: &Map<String, Value>) -> bool {
    let parts = filter.as_array().unwrap();
    if parts.len() == 2 {
        return match parts[0].as_str().unwrap() {
            "And" => parts[1]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| filter_matches(f, row)),
            "Or" => parts[1]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| filter_matches(f, row)),
            "Not" => !filter_matches(&parts[1], row),
            _ => false,
        };
    }
    let field = parts[0].as_str().unwrap();
    let left = row.get(field).unwrap_or(&Value::Null);
    let right = &parts[2];
    match parts[1].as_str().unwrap() {
        "Eq" => left == right,
        "NotEq" => left != right,
        "In" => right.as_array().unwrap().contains(left),
        "Contains" => left.as_array().is_some_and(|a| a.contains(right)),
        "NotContains" => left.as_array().is_some_and(|a| !a.contains(right)),
        "ContainsAny" => left
            .as_array()
            .is_some_and(|a| a.iter().any(|v| right.as_array().unwrap().contains(v))),
        "NotContainsAny" => left
            .as_array()
            .is_some_and(|a| a.iter().all(|v| !right.as_array().unwrap().contains(v))),
        "Gt" => !left.is_null() && compare_values(left, right).is_some_and(|o| o > 0),
        "Gte" => !left.is_null() && compare_values(left, right).is_some_and(|o| o >= 0),
        "Lt" => !left.is_null() && compare_values(left, right).is_some_and(|o| o < 0),
        "Lte" => left.is_null() || compare_values(left, right).is_some_and(|o| o <= 0),
        _ => false,
    }
}

fn compare_values(left: &Value, right: &Value) -> Option<i8> {
    if let (Some(a), Some(b)) = (left.as_bool(), right.as_bool()) {
        return Some(if a == b {
            0
        } else if a {
            1
        } else {
            -1
        });
    }
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
    if !parts.is_empty() && parts.iter().all(Value::is_array) {
        for part in parts {
            validate_rank(part, schema)?;
        }
        if parts.iter().all(is_attribute_order) {
            return Ok(());
        }
        return Err("multi-attribute rank requires attribute order clauses".into());
    }
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
            if !(field.starts_with("embed_")
                || field_type(&schema[field]).ends_with("]f16")
                || field_type(&schema[field]).ends_with("]f32"))
            {
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
fn is_attribute_order(rank: &Value) -> bool {
    rank.as_array().is_some_and(|parts| {
        (parts.len() == 2
            && parts[0].is_string()
            && matches!(parts[1].as_str(), Some("asc" | "desc")))
            || (!parts.is_empty()
                && parts.iter().all(|part| {
                    part.as_array().is_some_and(|item| {
                        item.len() == 2
                            && item[0].is_string()
                            && matches!(item[1].as_str(), Some("asc" | "desc"))
                    })
                }))
    })
}

fn compare_attribute_order(
    rank: &Value,
    left: &Map<String, Value>,
    right: &Map<String, Value>,
) -> std::cmp::Ordering {
    let parts = rank.as_array().unwrap();
    let clauses = if parts[0].is_array() {
        parts.iter().collect::<Vec<_>>()
    } else {
        vec![rank]
    };
    for clause in clauses {
        let clause = clause.as_array().unwrap();
        let field = clause[0].as_str().unwrap();
        let a = left.get(field).unwrap_or(&Value::Null);
        let b = right.get(field).unwrap_or(&Value::Null);
        let order = if a.is_null() {
            if b.is_null() {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Greater
            }
        } else if b.is_null() {
            std::cmp::Ordering::Less
        } else {
            compare_values(a, b).unwrap_or(0).cmp(&0)
        };
        let order = if clause[1] == "desc" {
            order.reverse()
        } else {
            order
        };
        if order != std::cmp::Ordering::Equal {
            return order;
        }
    }
    std::cmp::Ordering::Equal
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
    if is_attribute_order(rank) {
        return 0.0;
    }
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
