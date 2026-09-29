use crate::highlight;
use crate::store::{field_type, known_field, Namespace};
use crate::text::TextAnalysis;
use crate::vector;
use chrono::{DateTime, FixedOffset, NaiveDate};
use globset::GlobBuilder;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

impl Namespace {
    pub fn query(&self, body: &Value) -> Result<Value, String> {
        reset_text_caches();
        let mut normalized = body.clone();
        vector::normalize_query(&mut normalized, &self.schema)?;
        let mut result = self.query_inner(&normalized)?;
        if body.get("vector_encoding") == Some(&json!("base64")) {
            vector::encode_response(&mut result, &self.schema)?;
        }
        let bytes_queried = self.logical_bytes();
        let bytes_returned = serde_json::to_vec(&result).map_or(0, |bytes| bytes.len());
        result["billing"] = json!({
            "billable_logical_bytes_queried": bytes_queried,
            "billable_logical_bytes_returned": bytes_returned
        });
        result["performance"] = json!({
            "cache_hit_ratio": 1.0,
            "cache_temperature": "hot",
            "server_total_ms": 0,
            "query_execution_ms": 0,
            "exhaustive_search_count": self.rows.len(),
            "approx_namespace_size": self.rows.len()
        });
        Ok(result)
    }

    fn query_inner(&self, body: &Value) -> Result<Value, String> {
        let object = body.as_object().ok_or("query body must be an object")?;
        validate_query_options(object)?;
        if let Some(queries) = object.get("queries") {
            let queries = queries
                .as_array()
                .ok_or_else(|| crate::shape_error("queries must be an array"))?;
            if queries.is_empty() {
                return Err("must send at least one sub-query".into());
            }
            if queries.len() > 16 {
                return Err("multi-query exceeds the 16-subquery limit".into());
            }
            for query in queries {
                self.validate_query(query)?;
            }
            if object.contains_key("rerank_by") {
                let (rank_constant, weights, limit, offset) = validate_rrf(object, queries)?;
                let results = queries.iter().map(|q| self.execute_query(q));
                let mut fused = BTreeMap::<String, (f64, Value)>::new();
                for (result, weight) in results.zip(weights) {
                    for (index, row) in result["rows"].as_array().unwrap().iter().enumerate() {
                        let id = serde_json::to_string(&row["id"]).unwrap();
                        let entry = fused.entry(id).or_insert_with(|| (0.0, row.clone()));
                        entry.0 += weight / (rank_constant + index as f64 + 1.0);
                        if let (Some(existing), Some(current)) =
                            (entry.1.as_object_mut(), row.as_object())
                        {
                            for (key, value) in current {
                                if key != "$dist" {
                                    existing.insert(key.clone(), value.clone());
                                }
                            }
                        }
                    }
                }
                let mut rows = fused.into_values().collect::<Vec<_>>();
                rows.sort_by(|a, b| {
                    b.0.total_cmp(&a.0)
                        .then_with(|| compare_ids(&a.1["id"], &b.1["id"]))
                });
                let rows = rows
                    .into_iter()
                    .skip(offset)
                    .take(limit)
                    .map(|(score, mut row)| {
                        row["$dist"] = json!(serialized_f32(score as f32));
                        row
                    })
                    .collect::<Vec<_>>();
                return Ok(
                    json!({"results":[{"rows":rows}],"billing":{},"performance":{"server_total_ms":0}}),
                );
            }
            if object
                .keys()
                .any(|key| !matches!(key.as_str(), "queries" | "consistency" | "vector_encoding"))
            {
                return Err(
                    "queries cannot be combined with other query fields without rerank_by".into(),
                );
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
                    | "compute_attributes"
                    | "consistency"
                    | "distance_metric"
                    | "vector_encoding"
            ) {
                return Err(format!("unsupported query field {key}"));
            }
        }
        if let Some(rank) = object.get("rank_by") {
            validate_rank(rank, &self.schema)?;
            validate_nonnegative_rank(rank, &self.schema, false)?;
            if contains_knn(rank) && !object.contains_key("filters") {
                return Err("kNN requires filters".into());
            }
        }
        if let Some(filter) = object.get("filters") {
            validate_filter(filter, &self.schema)?;
        }
        if let Some(computed) = object.get("compute_attributes") {
            let computed = computed
                .as_object()
                .ok_or_else(|| crate::shape_error("compute_attributes must be an object"))?;
            if computed.len() > 256 {
                return Err("compute_attributes exceeds 256 fields".into());
            }
            for (name, expression) in computed {
                if name == "id" || name == "$dist" || known_field(&self.schema, name) {
                    return Err(format!(
                        "computed attribute {name} conflicts with an existing field"
                    ));
                }
                if highlight::is_highlight(expression) {
                    highlight::validate(name, expression, object.get("rank_by"), &self.schema)?;
                } else {
                    validate_computed(expression, &self.schema)?;
                }
            }
        }
        validate_query_options(object)?;
        if let Some(include) = object.get("include_attributes") {
            if !include.is_boolean() {
                let fields = include.as_array().ok_or_else(|| {
                    crate::shape_error("include_attributes must be an array or boolean")
                })?;
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
                .ok_or_else(|| crate::shape_error("exclude_attributes must be an array"))?;
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
        if object.get("top_k").is_some_and(|value| !value.is_number()) {
            return Err(crate::shape_error("top_k must be an integer"));
        }
        let limit = object.get("limit").or_else(|| object.get("top_k"));
        match limit {
            Some(Value::Number(n)) if n.as_u64().is_some_and(|v| v <= 10_000) => (),
            Some(Value::Object(v))
                if v.get("total")
                    .and_then(Value::as_u64)
                    .is_some_and(|n| n <= 10_000) =>
            {
                if v.keys().any(|key| !matches!(key.as_str(), "total" | "per")) {
                    return Err(crate::shape_error("unsupported limit field"));
                }
                if let Some(per) = v.get("per") {
                    let per = per.as_object().ok_or("limit.per must be an object")?;
                    if per
                        .keys()
                        .any(|key| !matches!(key.as_str(), "attributes" | "limit"))
                    {
                        return Err(crate::shape_error("unsupported limit.per field"));
                    }
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
            None => return Err("rank_by queries must specify top_k or limit".into()),
            Some(Value::Number(_)) => {
                return Err("limit or top_k must be an integer at most 10000".into());
            }
            _ => return Err(crate::shape_error("limit must be an integer or object")),
        }
        if object.get("offset").is_some_and(|v| v.as_u64().is_none()) {
            return Err("offset must be a nonnegative integer".into());
        }
        let total = limit
            .and_then(|value| value.as_u64().or_else(|| value["total"].as_u64()))
            .unwrap();
        let offset = object.get("offset").and_then(Value::as_u64).unwrap_or(0);
        if offset.saturating_add(total) > 10_000 {
            return Err("offset plus limit must be at most 10000".into());
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
        let metric = object
            .get("distance_metric")
            .and_then(Value::as_str)
            .or(self.distance_metric.as_deref())
            .unwrap_or("cosine_distance");
        let ascending = rank.is_none_or(|rank| is_ann(rank) || is_ascending(rank));
        let mut scored = self
            .rows
            .values()
            .filter_map(|row| {
                if filter.is_some_and(|f| !filter_matches(f, row, &self.schema)) {
                    return None;
                }
                let score = rank.map_or(0.0, |rank| {
                    score_rank(rank, row, &self.rows, &self.schema, metric)
                });
                let score = if score.is_finite() {
                    serialized_score(score)
                } else {
                    score
                };
                if !score.is_finite()
                    || rank.is_some_and(|rank| {
                        !is_ann(rank)
                            && !is_attribute_order(rank)
                            && !scores_every_row(rank)
                            && if has_max_floor(rank) {
                                !matches_rank(rank, row, &self.rows, &self.schema, metric)
                            } else {
                                score <= 0.0
                            }
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
                    .then_with(|| compare_ids(&row_a["id"], &row_b["id"]));
            }
            let order = a.total_cmp(b);
            (if ascending { order } else { order.reverse() })
                .then_with(|| compare_ids(&row_a["id"], &row_b["id"]))
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
                    result.insert("$dist".into(), json!(serialized_score(score)));
                }
                if let Some(computed) = object.get("compute_attributes").and_then(Value::as_object)
                {
                    for (name, expression) in computed {
                        let value = if highlight::is_highlight(expression) {
                            highlight::compute(expression, rank, row, &self.schema, metric)
                        } else {
                            computed_value(expression, row, &self.rows, &self.schema, metric)
                        };
                        result.insert(name.clone(), value);
                    }
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
                "aggregate_by"
                    | "group_by"
                    | "filters"
                    | "top_k"
                    | "vector_encoding"
                    | "consistency"
            ) {
                return Err(crate::shape_error(format!(
                    "unknown aggregation field {key}"
                )));
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
                _ => return Err(crate::shape_error(format!("unsupported aggregate {name}"))),
            }
        }
        if let Some(filter) = object.get("filters") {
            validate_filter(filter, &self.schema)?;
        }
        if let Some(groups) = object.get("group_by") {
            let groups = groups
                .as_array()
                .ok_or_else(|| crate::shape_error("group_by must be an array"))?;
            let mut seen = std::collections::HashSet::new();
            let mut has_for_each_unique = false;
            for entry in groups {
                let (name, source, explode) = if let Some(field) = entry.as_str() {
                    (field, field, false)
                } else {
                    let expression = entry.as_object().ok_or_else(|| {
                        crate::shape_error("group_by entry must be an attribute or expression")
                    })?;
                    if expression.len() != 1 {
                        return Err(crate::shape_error("group_by expression needs one name"));
                    }
                    let (alias, expression) = expression.iter().next().unwrap();
                    let parts = expression.as_array().ok_or_else(|| {
                        crate::shape_error("group_by expression must be an array")
                    })?;
                    if parts.len() != 2 || parts[0] != "ForEachUnique" {
                        return Err(crate::shape_error("unsupported group_by expression"));
                    }
                    let source = parts[1].as_str().ok_or_else(|| {
                        crate::shape_error("ForEachUnique attribute must be a string")
                    })?;
                    (alias.as_str(), source, true)
                };
                if name == "id" {
                    return Err("cannot group by id".into());
                }
                if !seen.insert(name) {
                    return Err(format!("cannot use {name} in group_by more than once"));
                }
                if !known_field(&self.schema, source) {
                    return Err(format!("attribute {source} does not exist in schema"));
                }
                if explode
                    && !self.schema.get(source).is_some_and(|definition| {
                        matches!(
                            field_type(definition),
                            "[]string"
                                | "[]int"
                                | "[]uint"
                                | "[]float"
                                | "[]bool"
                                | "[]datetime"
                                | "[]uuid"
                        )
                    })
                {
                    return Err(format!(
                        "ForEachUnique requires an array attribute {source}"
                    ));
                }
                if explode {
                    if has_for_each_unique {
                        return Err("cannot use ForEachUnique more than once in group_by".into());
                    }
                    has_for_each_unique = true;
                }
                if aggregates.contains_key(name) {
                    return Err(format!(
                        "aggregate name {name} conflicts with a group field"
                    ));
                }
            }
        }
        if object.contains_key("top_k") && !object.contains_key("group_by") {
            return Err("top_k requires group_by for aggregation".into());
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
                    .is_none_or(|filter| filter_matches(filter, row, &self.schema))
            })
            .collect::<Vec<_>>();
        let aggregates = object["aggregate_by"].as_object().unwrap();
        let fields = object.get("group_by").and_then(Value::as_array);
        let Some(fields) = fields.filter(|fields| !fields.is_empty()) else {
            return json!({"aggregations": aggregate_values(aggregates, &rows, &self.schema)});
        };
        let fields = group_fields(fields);
        let mut grouped = std::collections::BTreeMap::<
            String,
            (Map<String, Value>, Vec<&Map<String, Value>>),
        >::new();
        for row in rows {
            let mut combinations = vec![Vec::new()];
            for field in &fields {
                let values = if field.explode {
                    match row.get(field.source) {
                        Some(Value::Array(values)) => {
                            let mut distinct = Vec::new();
                            for value in values {
                                if !distinct.contains(value) {
                                    distinct.push(value.clone());
                                }
                            }
                            distinct
                        }
                        _ => vec![Value::Null],
                    }
                } else {
                    vec![row.get(field.source).cloned().unwrap_or(Value::Null)]
                };
                combinations = combinations
                    .into_iter()
                    .flat_map(|prefix| {
                        values.iter().map(move |value| {
                            let mut group = prefix.clone();
                            group.push(value.clone());
                            group
                        })
                    })
                    .collect();
            }
            for values in combinations {
                let key = serde_json::to_string(&values).unwrap();
                let entry = grouped.entry(key).or_insert_with(|| {
                    let attrs = fields
                        .iter()
                        .zip(&values)
                        .map(|(field, value)| (field.name.to_owned(), value.clone()))
                        .collect();
                    (attrs, Vec::new())
                });
                entry.1.push(row);
            }
        }
        let limit = object
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(10_000) as usize;
        let mut groups = grouped.into_values().collect::<Vec<_>>();
        groups.sort_by(|(left, _), (right, _)| {
            for field in &fields {
                let left = left.get(field.name).unwrap_or(&Value::Null);
                let right = right.get(field.name).unwrap_or(&Value::Null);
                let order = match (left.is_null(), right.is_null()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => compare_values(left, right).unwrap_or(0).cmp(&0),
                };
                if order != std::cmp::Ordering::Equal {
                    return order;
                }
            }
            std::cmp::Ordering::Equal
        });
        let groups = groups
            .into_iter()
            .take(limit)
            .map(|(mut attrs, rows)| {
                attrs.extend(aggregate_values(aggregates, &rows, &self.schema));
                Value::Object(attrs)
            })
            .collect::<Vec<_>>();
        json!({"aggregation_groups":groups})
    }
}

struct GroupField<'a> {
    name: &'a str,
    source: &'a str,
    explode: bool,
}

fn group_fields(fields: &[Value]) -> Vec<GroupField<'_>> {
    fields
        .iter()
        .map(|field| {
            if let Some(name) = field.as_str() {
                GroupField {
                    name,
                    source: name,
                    explode: false,
                }
            } else {
                let (name, expression) = field.as_object().unwrap().iter().next().unwrap();
                GroupField {
                    name,
                    source: expression[1].as_str().unwrap(),
                    explode: true,
                }
            }
        })
        .collect()
}

fn validate_query_options(object: &Map<String, Value>) -> Result<(), String> {
    if object
        .get("vector_encoding")
        .is_some_and(|encoding| !matches!(encoding.as_str(), Some("float" | "base64")))
    {
        if object["vector_encoding"].is_string() {
            return Err(crate::shape_error(
                "vector_encoding must be float or base64",
            ));
        }
        return Err("vector_encoding must be float or base64".into());
    }
    if let Some(consistency) = object.get("consistency") {
        let consistency = consistency
            .as_object()
            .ok_or_else(|| crate::shape_error("consistency must be an object"))?;
        if !matches!(
            consistency.get("level").and_then(Value::as_str),
            Some("strong" | "eventual")
        ) {
            return Err(crate::shape_error(
                "consistency.level must be strong or eventual",
            ));
        }
    }
    if object.get("distance_metric").is_some_and(|value| {
        !matches!(
            value.as_str(),
            Some("cosine_distance" | "euclidean_squared")
        )
    }) {
        return Err(crate::shape_error(
            "distance_metric must be cosine_distance or euclidean_squared",
        ));
    }
    Ok(())
}

fn validate_computed(expression: &Value, schema: &Map<String, Value>) -> Result<(), String> {
    let parts = expression
        .as_array()
        .ok_or_else(|| crate::shape_error("computed attribute expression must be an array"))?;
    if parts.get(1) == Some(&json!("BM25")) {
        return validate_rank(expression, schema);
    }
    if parts.len() != 3 {
        return Err(crate::shape_error(
            "computed attribute requires a three-part expression",
        ));
    }
    if parts[1] == "VectorDist" {
        let mut rank = parts.clone();
        rank[1] = json!("ANN");
        return validate_rank(&Value::Array(rank), schema);
    }
    Err("only BM25 and VectorDist computed attributes are supported".into())
}

fn computed_value(
    expression: &Value,
    row: &Map<String, Value>,
    corpus: &BTreeMap<String, Map<String, Value>>,
    schema: &Map<String, Value>,
    metric: &str,
) -> Value {
    if expression[1] == "BM25" {
        let parts = expression.as_array().unwrap();
        let field = parts[0].as_str().unwrap();
        let last_as_prefix = parts
            .get(3)
            .and_then(|options| options.get("last_as_prefix"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return json!(bm25(
            field,
            &parts[2],
            last_as_prefix,
            row,
            corpus,
            schema,
            true,
        ));
    }
    let mut rank = expression.clone();
    if rank[1] == "VectorDist" {
        rank[1] = json!("ANN");
    }
    let score = score_rank(&rank, row, corpus, schema, metric);
    if score.is_finite() {
        json!(serialized_score(score))
    } else {
        Value::Null
    }
}

fn validate_rrf(
    object: &Map<String, Value>,
    queries: &[Value],
) -> Result<(f64, Vec<f64>, usize, usize), String> {
    if queries.len() < 2 || queries.iter().any(|q| q.get("aggregate_by").is_some()) {
        return Err("RRF requires at least two non-aggregation queries".into());
    }
    if object.keys().any(|key| {
        !matches!(
            key.as_str(),
            "queries" | "rerank_by" | "limit" | "offset" | "consistency" | "vector_encoding"
        )
    }) {
        return Err("unsupported multi-query field".into());
    }
    let parts = object["rerank_by"]
        .as_array()
        .ok_or_else(|| crate::shape_error("rerank_by must be an array"))?;
    if parts.is_empty() || parts.len() > 2 || parts[0] != "RRF" {
        return Err("only RRF reranking is supported".into());
    }
    let config = match parts.get(1) {
        Some(value) => Some(value.as_object().ok_or("RRF config must be an object")?),
        None => None,
    };
    if config.is_some_and(|config| {
        config
            .keys()
            .any(|key| !matches!(key.as_str(), "weights" | "rank_constant"))
    }) {
        return Err("unsupported RRF config field".into());
    }
    let rank_constant = config
        .and_then(|c| c.get("rank_constant"))
        .map_or(Ok(60), |v| {
            v.as_u64()
                .filter(|v| *v > 0)
                .ok_or("rank_constant must be a positive integer")
        })? as f64;
    let weights = if let Some(values) = config.and_then(|c| c.get("weights")) {
        let values = values.as_array().ok_or("weights must be an array")?;
        if values.len() != queries.len() {
            return Err("weights must match the number of queries".into());
        }
        values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|weight| weight.is_finite() && *weight > 0.0)
                    .ok_or("weights must be positive numbers".into())
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        vec![1.0; queries.len()]
    };
    let limit_value = object.get("limit").ok_or("RRF requires a limit")?;
    let limit = match limit_value {
        Value::Number(number) => number.as_u64(),
        Value::Object(config) if config.len() == 1 => config.get("total").and_then(Value::as_u64),
        _ => None,
    }
    .ok_or("RRF limit must be an integer or {total: integer}")?;
    if limit > 10_000 {
        return Err("RRF limit must be at most 10000".into());
    }
    let offset = object.get("offset").map_or(Ok(0), |v| {
        v.as_u64().ok_or("offset must be a nonnegative integer")
    })?;
    if offset.saturating_add(limit) > 10_000 {
        return Err("offset plus limit must be at most 10000".into());
    }
    Ok((rank_constant, weights, limit as usize, offset as usize))
}

fn aggregate_values(
    aggregates: &Map<String, Value>,
    rows: &[&Map<String, Value>],
    schema: &Map<String, Value>,
) -> Map<String, Value> {
    aggregates
        .iter()
        .map(|(name, expression)| {
            let parts = expression.as_array().unwrap();
            let value = if parts[0] == "Count" {
                json!(rows.len())
            } else {
                let field = parts[1].as_str().unwrap();
                if field_type(&schema[field]) == "float" {
                    let total: f64 = rows
                        .iter()
                        .filter_map(|row| row.get(field).and_then(Value::as_f64))
                        .sum();
                    json!(total)
                } else {
                    let total: i128 = rows
                        .iter()
                        .filter_map(|row| row.get(field))
                        .filter_map(|value| {
                            value
                                .as_i64()
                                .map(i128::from)
                                .or_else(|| value.as_u64().map(i128::from))
                        })
                        .sum();
                    if let Ok(signed) = i64::try_from(total) {
                        json!(signed)
                    } else if let Ok(unsigned) = u64::try_from(total) {
                        json!(unsigned)
                    } else {
                        json!(total as f64)
                    }
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
    if parts.len() != 3 && parts.len() != 4 {
        return Err("filter must have three or four elements".into());
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
    let definition = schema
        .get(field)
        .ok_or(format!("attribute {field} is not filterable"))?;
    if matches!(field_type(definition), "bytes" | "{}f16")
        || vector::multi_dimensions(definition).is_some()
    {
        return Err(format!("attribute {field} is not filterable"));
    }
    let full_text_search = has_full_text_search(definition);
    let filterable = definition
        .get("filterable")
        .and_then(Value::as_bool)
        .unwrap_or(
            !full_text_search
                && definition.get("fuzzy") != Some(&Value::Bool(true))
                && definition.get("regex") != Some(&Value::Bool(true))
                && definition.get("glob") != Some(&Value::Bool(true)),
        );
    if !filterable
        && !matches!(
            op,
            "ContainsAllTokens"
                | "ContainsAnyToken"
                | "ContainsTokenSequence"
                | "Glob"
                | "NotGlob"
                | "IGlob"
                | "NotIGlob"
                | "Regex"
                | "Fuzzy"
        )
    {
        return Err(format!("attribute {field} is not filterable"));
    }
    if matches!(
        op,
        "Contains" | "NotContains" | "ContainsAny" | "NotContainsAny"
    ) && !schema
        .get(field)
        .is_some_and(|definition| field_type(definition).starts_with("[]"))
    {
        return Err(format!("attribute {field} is not an array"));
    }
    if matches!(
        op,
        "ContainsAllTokens" | "ContainsAnyToken" | "ContainsTokenSequence"
    ) {
        if !matches!(field_type(&schema[field]), "string" | "[]string") {
            return Err(format!("attribute {field} is not a text field"));
        }
        if !schema.get(field).is_some_and(has_full_text_search) {
            return Err(format!(
                "attribute {field} is not configured for full-text search"
            ));
        }
        let pre_tokenized = TextAnalysis::for_field(&schema[field])
            .is_some_and(|analysis| analysis.is_pre_tokenized());
        let string_array = parts[2]
            .as_array()
            .is_some_and(|tokens| tokens.iter().all(Value::is_string));
        if pre_tokenized && !string_array {
            return Err(format!(
                "filter error in key `{field}`: type mismatch, {op} expects []string, but got {} (cannot cast {} into type []string)",
                json_type(&parts[2]),
                json_type(&parts[2])
            ));
        }
        if !parts[2].is_string() && !string_array {
            return Err(format!("{op} requires a string or string array"));
        }
        if parts.len() == 4 {
            if op == "ContainsTokenSequence" {
                return Err("ContainsTokenSequence does not take options".into());
            }
            let options = parts[3]
                .as_object()
                .ok_or("token filter options must be an object")?;
            if options.keys().any(|key| key != "last_as_prefix")
                || options
                    .get("last_as_prefix")
                    .is_some_and(|value| !value.is_boolean())
            {
                return Err("only boolean last_as_prefix is supported".into());
            }
        }
        return Ok(());
    }
    if matches!(op, "Glob" | "NotGlob" | "IGlob" | "NotIGlob" | "Regex") {
        let kind = field_type(&schema[field]);
        if parts.len() != 3
            || (op == "Regex" && kind != "string")
            || (op != "Regex" && !matches!(kind, "string" | "[]string"))
        {
            return Err(format!(
                "{op} requires a string{} attribute and a three-part filter",
                if op == "Regex" {
                    ""
                } else {
                    " or string-array"
                }
            ));
        }
        let capability = if op == "Regex" { "regex" } else { "glob" };
        if schema[field].get(capability) != Some(&Value::Bool(true)) {
            return Err(format!(
                "attribute {field} does not enable {capability} filters"
            ));
        }
        let pattern = parts[2].as_str().ok_or("pattern must be a string")?;
        if op == "Regex" {
            Regex::new(pattern).map_err(|error| format!("invalid regex: {error}"))?;
        } else {
            GlobBuilder::new(pattern)
                .case_insensitive(matches!(op, "IGlob" | "NotIGlob"))
                .build()
                .map_err(|error| format!("invalid glob: {error}"))?;
        }
        return Ok(());
    }
    if op == "Fuzzy" {
        if parts.len() != 4
            || !matches!(field_type(definition), "string" | "[]string")
            || definition.get("fuzzy") != Some(&Value::Bool(true))
            || !parts[2].is_string()
        {
            return Err(format!(
                "attribute {field} does not enable fuzzy text filtering"
            ));
        }
        let options = parts[3]
            .as_object()
            .ok_or("Fuzzy options must be an object")?;
        if options
            .keys()
            .any(|key| !matches!(key.as_str(), "case_sensitive" | "max_edit_distance"))
            || options
                .get("case_sensitive")
                .is_some_and(|value| !value.is_boolean())
        {
            return Err("invalid Fuzzy options".into());
        }
        let thresholds = options
            .get("max_edit_distance")
            .and_then(Value::as_array)
            .ok_or("Fuzzy requires max_edit_distance thresholds")?;
        if thresholds.is_empty()
            || thresholds.iter().any(|threshold| {
                threshold.as_object().is_none_or(|entry| {
                    entry.len() != 2
                        || entry
                            .get("min_query_chars")
                            .and_then(Value::as_u64)
                            .is_none()
                        || entry
                            .get("distance")
                            .and_then(Value::as_u64)
                            .is_none_or(|distance| {
                                distance > 2
                                    || entry["min_query_chars"].as_u64().unwrap()
                                        < 3 * (distance + 1)
                            })
                })
            })
        {
            return Err("invalid Fuzzy max_edit_distance thresholds".into());
        }
        return Ok(());
    }
    if parts.len() == 4 {
        return Err(format!("{op} does not take options"));
    }
    match op {
        "Eq" | "NotEq" | "Gt" | "Gte" | "Lt" | "Lte" | "Contains" | "NotContains" => Ok(()),
        "In" | "NotIn"
            if parts[2]
                .as_array()
                .is_some_and(|values| values.iter().any(Value::is_null)) =>
        {
            Err(crate::shape_error(format!(
                "{op} values cannot include null"
            )))
        }
        "In" | "NotIn" | "ContainsAny" | "NotContainsAny" if parts[2].is_array() => Ok(()),
        "In" | "NotIn" | "ContainsAny" | "NotContainsAny" => Err(format!("{op} requires an array")),
        "AnyGt" | "AnyGte" | "AnyLt" | "AnyLte"
            if schema
                .get(field)
                .is_some_and(|definition| field_type(definition).starts_with("[]")) =>
        {
            Ok(())
        }
        "AnyGt" | "AnyGte" | "AnyLt" | "AnyLte" => {
            Err(format!("attribute {field} is not an array"))
        }
        _ => Err(crate::shape_error(format!(
            "unsupported filter operator {op}"
        ))),
    }
}

fn has_full_text_search(definition: &Value) -> bool {
    definition
        .get("full_text_search")
        .is_some_and(|config| config == &Value::Bool(true) || config.is_object())
}

pub(crate) fn filter_matches(
    filter: &Value,
    row: &Map<String, Value>,
    schema: &Map<String, Value>,
) -> bool {
    let parts = filter.as_array().unwrap();
    if parts.len() == 2 {
        return match parts[0].as_str().unwrap() {
            "And" => parts[1]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| filter_matches(f, row, schema)),
            "Or" => parts[1]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| filter_matches(f, row, schema)),
            "Not" => !filter_matches(&parts[1], row, schema),
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
        "NotIn" => !right.as_array().unwrap().contains(left),
        "Contains" => left.as_array().is_some_and(|a| a.contains(right)),
        "NotContains" => left.as_array().is_none_or(|a| !a.contains(right)),
        "ContainsAny" => left
            .as_array()
            .is_some_and(|a| a.iter().any(|v| right.as_array().unwrap().contains(v))),
        "NotContainsAny" => left
            .as_array()
            .is_none_or(|a| a.iter().all(|v| !right.as_array().unwrap().contains(v))),
        "Gt" => {
            if right.is_null() {
                !left.is_null()
            } else {
                !left.is_null() && compare_values(left, right).is_some_and(|o| o > 0)
            }
        }
        "Gte" => {
            right.is_null()
                || (!left.is_null() && compare_values(left, right).is_some_and(|o| o >= 0))
        }
        "Lt" => {
            !right.is_null()
                && (left.is_null() || compare_values(left, right).is_some_and(|o| o < 0))
        }
        "Lte" => {
            if right.is_null() {
                left.is_null()
            } else {
                left.is_null() || compare_values(left, right).is_some_and(|o| o <= 0)
            }
        }
        "AnyGt" | "AnyGte" | "AnyLt" | "AnyLte" => left.as_array().is_some_and(|values| {
            values.iter().any(|value| match parts[1].as_str().unwrap() {
                "AnyGt" => compare_values(value, right).is_some_and(|order| order > 0),
                "AnyGte" => compare_values(value, right).is_some_and(|order| order >= 0),
                "AnyLt" => compare_values(value, right).is_some_and(|order| order < 0),
                _ => compare_values(value, right).is_some_and(|order| order <= 0),
            })
        }),
        "ContainsAllTokens" | "ContainsAnyToken" | "ContainsTokenSequence" => {
            token_filter_matches(parts, left, schema.get(field).unwrap_or(&Value::Null))
        }
        "Glob" | "NotGlob" | "IGlob" | "NotIGlob" => {
            let matcher = GlobBuilder::new(right.as_str().unwrap())
                .case_insensitive(matches!(parts[1].as_str(), Some("IGlob" | "NotIGlob")))
                .build()
                .unwrap()
                .compile_matcher();
            let matches = match left {
                Value::String(text) => matcher.is_match(text),
                Value::Array(values) => values
                    .iter()
                    .any(|value| value.as_str().is_some_and(|text| matcher.is_match(text))),
                _ => false,
            };
            if matches!(parts[1].as_str(), Some("NotGlob" | "NotIGlob")) {
                !matches
            } else {
                matches
            }
        }
        "Regex" => left
            .as_str()
            .is_some_and(|text| Regex::new(right.as_str().unwrap()).unwrap().is_match(text)),
        "Fuzzy" => fuzzy_matches(left, right.as_str().unwrap(), &parts[3]),
        _ => false,
    }
}

fn fuzzy_matches(value: &Value, query: &str, options: &Value) -> bool {
    let max_distance = options["max_edit_distance"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| {
            query.chars().count() >= entry["min_query_chars"].as_u64().unwrap() as usize
        })
        .filter_map(|entry| entry["distance"].as_u64())
        .max();
    let Some(max_distance) = max_distance else {
        return false;
    };
    let case_sensitive = options
        .get("case_sensitive")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let query = if case_sensitive {
        query.to_owned()
    } else {
        query.to_lowercase()
    };
    let candidates: Vec<&str> = match value {
        Value::String(text) => vec![text],
        Value::Array(values) => values.iter().filter_map(Value::as_str).collect(),
        _ => return false,
    };
    candidates.into_iter().any(|candidate| {
        let candidate = if case_sensitive {
            candidate.to_owned()
        } else {
            candidate.to_lowercase()
        };
        // A match needs at least `query - distance` characters of text, so shorter
        // values are skipped before the quadratic distance computation.
        candidate.chars().count() + max_distance as usize >= query.chars().count()
            && fuzzy_substring_within(&query, &candidate, max_distance as usize)
    })
}

/// Whether some substring of `text` is within `max_distance` edits of `query`. The scan
/// stops at the first such substring, and reuses two rows instead of allocating per
/// character.
fn fuzzy_substring_within(query: &str, text: &str, max_distance: usize) -> bool {
    let query: Vec<char> = query.chars().collect();
    if query.len() <= max_distance {
        return true;
    }
    let mut previous: Vec<usize> = (0..=query.len()).collect();
    let mut current = vec![0; query.len() + 1];
    for character in text.chars() {
        for (index, expected) in query.iter().enumerate() {
            current[index + 1] = (previous[index + 1] + 1)
                .min(current[index] + 1)
                .min(previous[index] + usize::from(*expected != character));
        }
        if current[query.len()] <= max_distance {
            return true;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    false
}

fn token_filter_matches(parts: &[Value], left: &Value, definition: &Value) -> bool {
    let Some(analysis) = TextAnalysis::for_field(definition) else {
        return false;
    };
    let document = analysis.analyze(left);
    let query = analysis.analyze(&parts[2]);
    if document.is_empty() || query.is_empty() {
        return false;
    }
    let last_as_prefix = parts
        .get(3)
        .and_then(|options| options.get("last_as_prefix"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let matches_term = |index: usize, token: &str| {
        document.iter().any(|candidate| {
            if last_as_prefix && index + 1 == query.len() {
                candidate.text.starts_with(token)
            } else {
                candidate.text == token
            }
        })
    };
    match parts[1].as_str().unwrap() {
        "ContainsAllTokens" => query
            .iter()
            .enumerate()
            .all(|(index, token)| matches_term(index, &token.text)),
        "ContainsAnyToken" => query
            .iter()
            .enumerate()
            .any(|(index, token)| matches_term(index, &token.text)),
        // Positions keep the gaps a stopword leaves, on both sides, so "visited the café"
        // matches "visited a café" once stopwords are removed.
        "ContainsTokenSequence" => {
            let first = query[0].position;
            let at: HashMap<(usize, &str), ()> = document
                .iter()
                .map(|token| ((token.position, token.text.as_str()), ()))
                .collect();
            document.iter().any(|start| {
                start.text == query[0].text
                    && query.iter().all(|token| {
                        at.contains_key(&(
                            start.position + token.position - first,
                            token.text.as_str(),
                        ))
                    })
            })
        }
        _ => false,
    }
}

fn compare_ids(left: &Value, right: &Value) -> std::cmp::Ordering {
    match (left.as_u64(), right.as_u64()) {
        (Some(left), Some(right)) => left.cmp(&right),
        _ => left.as_str().cmp(&right.as_str()),
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

pub(crate) fn validate_rank(rank: &Value, schema: &Map<String, Value>) -> Result<(), String> {
    let parts = rank
        .as_array()
        .ok_or_else(|| crate::shape_error("rank_by must be an array"))?;
    if !parts.is_empty() && parts.iter().all(Value::is_array) {
        for part in parts {
            validate_rank(part, schema)?;
        }
        if parts.iter().all(is_attribute_order) {
            return Ok(());
        }
        return Err("multi-attribute rank requires attribute order clauses".into());
    }
    if parts.len() == 2 && matches!(parts[0].as_str(), Some("Sum" | "Max")) {
        let children = parts[1].as_array().ok_or("Sum/Max requires an array")?;
        if children.is_empty() {
            return Err("Sum/Max requires at least one expression".into());
        }
        for child in children {
            // Max accepts scalar floors such as ["Max", [0, expression]]; Sum does not.
            if parts[0] == "Sum" && child.is_number() {
                return Err("Sum aggregations can only contain other text queries".into());
            }
            if parts[0] == "Max" && child.is_number() {
                if child
                    .as_f64()
                    .is_none_or(|number| !number.is_finite() || number < 0.0)
                {
                    return Err(format!(
                        "Scalar values in rank_by must be non-negative, got {child}"
                    ));
                }
                continue;
            }
            validate_rank(child, schema)?;
        }
        return Ok(());
    }
    if parts.len() == 3 && matches!(parts[0].as_str(), Some("Saturate" | "Decay")) {
        validate_rank(&parts[1], schema)?;
        let config = parts[2]
            .as_object()
            .ok_or("Saturate/Decay requires a config object")?;
        if config
            .keys()
            .any(|key| !matches!(key.as_str(), "midpoint" | "exponent"))
        {
            return Err("unsupported Saturate/Decay option".into());
        }
        let midpoint = config
            .get("midpoint")
            .ok_or("Saturate/Decay requires midpoint")?;
        if midpoint
            .as_f64()
            .is_none_or(|number| !number.is_finite() || number <= 0.0)
            && (midpoint.as_str().and_then(parse_duration_ms).is_none()
                || !is_datetime_dist(&parts[1], schema))
        {
            return Err("Saturate/Decay midpoint must be positive".into());
        }
        if config.get("exponent").is_some_and(|exponent| {
            exponent
                .as_f64()
                .is_none_or(|number| !number.is_finite() || number <= 0.0)
        }) {
            return Err("Saturate/Decay exponent must be positive".into());
        }
        return Ok(());
    }
    if parts.len() == 3 && parts[0] == "Dist" {
        let attribute = parts[1]
            .as_array()
            .ok_or("Dist requires an Attribute expression")?;
        if attribute.len() != 2 || attribute[0] != "Attribute" {
            return Err("Dist requires an Attribute expression".into());
        }
        let field = attribute[1].as_str().ok_or("Dist requires a field name")?;
        let definition = schema.get(field).ok_or("Dist attribute does not exist")?;
        let valid = match field_type(definition) {
            "uint" => parts[2].is_u64(),
            "int" => parts[2].is_i64() || parts[2].is_u64(),
            "float" => parts[2].is_number(),
            "datetime" => parts[2].as_str().and_then(parse_datetime_ms).is_some(),
            _ => false,
        };
        if !valid {
            return Err("Dist origin must match a numeric or datetime attribute".into());
        }
        return Ok(());
    }
    if parts.len() == 3 && parts[0] == "Product" {
        let (weight, expression) = if parts[1].is_number() {
            (&parts[1], &parts[2])
        } else {
            (&parts[2], &parts[1])
        };
        if !weight
            .as_f64()
            .is_some_and(|value| value.is_finite() && value >= 0.0)
        {
            return Err("Product requires a nonnegative numeric weight".into());
        }
        return validate_rank(expression, schema);
    }
    if parts.len() == 2 && parts[0] == "Attribute" {
        let field = parts[1].as_str().ok_or("Attribute requires a field name")?;
        if !known_field(schema, field)
            || !matches!(field_type(&schema[field]), "uint" | "int" | "float")
        {
            return Err(format!("attribute {field} is not numeric"));
        }
        return Ok(());
    }
    if (parts.len() == 3 || parts.len() == 4)
        && parts[0].is_string()
        && parts[1].as_str().is_some_and(is_filter_operator)
    {
        return validate_filter(rank, schema);
    }
    if parts.len() != 4 && parts.len() != 3 && parts.len() != 2 {
        return Err("unsupported rank_by expression".into());
    }
    let field = parts[0].as_str().ok_or("rank attribute must be a string")?;
    if !known_field(schema, field) {
        return Err(format!("attribute {field} does not exist in schema"));
    }
    let operator = parts[1].as_str().ok_or("rank operator must be a string")?;
    match operator {
        "SparseKNN" if parts.len() == 3 => {
            if schema
                .get(field)
                .is_none_or(|definition| field_type(definition) != "{}f16")
                || schema
                    .get(field)
                    .and_then(|definition| definition.get("sparse_knn"))
                    .and_then(|config| config.get("distance_metric"))
                    != Some(&json!("dot_product"))
            {
                return Err(format!("attribute {field} does not enable SparseKNN"));
            }
            let sparse = parts[2]
                .as_object()
                .ok_or("SparseKNN query must be an object")?;
            if sparse.len() > 1024 || sparse.values().any(|value| !value.is_number()) {
                return Err("SparseKNN query must contain at most 1024 numeric dimensions".into());
            }
            Ok(())
        }
        "ANN" | "kNN"
            if parts.len() == 3
                && schema
                    .get(field)
                    .and_then(vector::multi_dimensions)
                    .is_some() =>
        {
            let dimensions = schema
                .get(field)
                .and_then(vector::multi_dimensions)
                .unwrap();
            let vectors = parts[2]
                .as_array()
                .ok_or("multi-vector query must be an array")?;
            if vectors.is_empty()
                || !vector::multi_vector_within_limit(vectors.len(), dimensions)
                || vectors.iter().any(|vector| {
                    vector.as_array().is_none_or(|elements| {
                        elements.len() != dimensions
                            || elements.iter().any(|value| !value.is_number())
                    })
                })
            {
                return Err(format!(
                    "query vectors have wrong dimensions for attribute {field}"
                ));
            }
            if operator == "ANN"
                && schema[field]
                    .get("ann")
                    .and_then(|ann| ann.get("late_interaction"))
                    != Some(&Value::Bool(true))
            {
                return Err(format!(
                    "attribute {field} does not enable late-interaction ANN"
                ));
            }
            Ok(())
        }
        "ANN" | "kNN"
            if parts.len() == 3
                && parts[2]
                    .as_array()
                    .is_some_and(|a| a.iter().all(Value::is_number)) =>
        {
            if !(field.starts_with("embed_") || vector::dimensions(&schema[field]).is_some()) {
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
            if operator == "ANN"
                && !field.starts_with("embed_")
                && schema[field].get("ann") != Some(&Value::Bool(true))
            {
                return Err(format!("attribute {field} does not enable ANN"));
            }
            Ok(())
        }
        "BM25" if parts.len() == 3 || parts.len() == 4 => {
            let definition = schema.get(field).unwrap_or(&Value::Null);
            if !matches!(field_type(definition), "string" | "[]string")
                || !has_full_text_search(definition)
            {
                return Err(format!("attribute {field} has no full-text index"));
            }
            let pre_tokenized = TextAnalysis::for_field(definition)
                .is_some_and(|analysis| analysis.is_pre_tokenized());
            let operand_ok = if pre_tokenized {
                parts[2]
                    .as_array()
                    .is_some_and(|tokens| tokens.iter().all(Value::is_string))
            } else {
                parts[2].is_string()
            };
            if !operand_ok {
                return Err(format!(
                    "invalid input of type {} for rank_by field \"{field}\", expecting {}",
                    json_type(&parts[2]),
                    if pre_tokenized { "[]string" } else { "string" }
                ));
            }
            if parts.len() == 4 {
                let options = parts[3]
                    .as_object()
                    .ok_or("BM25 options must be an object")?;
                if options.len() != 1
                    || !options.get("last_as_prefix").is_some_and(Value::is_boolean)
                {
                    return Err("only boolean last_as_prefix is supported for BM25".into());
                }
            }
            Ok(())
        }
        "asc" | "desc"
            if parts.len() == 2
                && schema.get(field).is_some_and(|definition| {
                    !matches!(field_type(definition), "bytes" | "{}f16")
                        && vector::multi_dimensions(definition).is_none()
                }) =>
        {
            Ok(())
        }
        "asc" | "desc" if parts.len() == 2 => {
            Err(format!("attribute {field} cannot be used for ordering"))
        }
        _ => Err(crate::shape_error(format!(
            "unsupported rank operator {operator}"
        ))),
    }
}

/// Attribute-derived clauses score every row, including rows that score zero. Text,
/// filter and sparse clauses only return rows that match.
fn scores_every_row(rank: &Value) -> bool {
    rank.as_array().is_some_and(|parts| {
        (parts.len() >= 2 && matches!(parts[0].as_str(), Some("Attribute" | "Dist")))
            || parts.iter().any(scores_every_row)
    })
}

/// Whether a `Max` list somewhere in `rank` carries a scalar floor.
fn has_max_floor(rank: &Value) -> bool {
    rank.as_array().is_some_and(|parts| {
        (parts.len() == 2
            && parts[0] == "Max"
            && parts[1]
                .as_array()
                .is_some_and(|children| children.iter().any(Value::is_number)))
            || parts.iter().any(has_max_floor)
    })
}

/// A scalar floor raises the score of every row, but only rows that match one of the
/// expression's clauses are returned.
fn matches_rank(
    rank: &Value,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
    schema: &Map<String, Value>,
    metric: &str,
) -> bool {
    let parts = rank.as_array().unwrap();
    match parts[0].as_str() {
        Some("Sum" | "Max") if parts.len() == 2 => parts[1]
            .as_array()
            .unwrap()
            .iter()
            .filter(|child| !child.is_number())
            .any(|child| matches_rank(child, row, corpus, schema, metric)),
        Some("Product") if parts.len() == 3 => {
            let expression = if parts[1].is_number() {
                &parts[2]
            } else {
                &parts[1]
            };
            matches_rank(expression, row, corpus, schema, metric)
        }
        Some("Saturate" | "Decay") => matches_rank(&parts[1], row, corpus, schema, metric),
        _ => score_rank(rank, row, corpus, schema, metric) > 0.0,
    }
}

/// Turbopuffer rejects a ranking score that can be negative. A signed attribute must
/// sit under a `Max` with a scalar floor, or feed a clause that clamps it.
fn validate_nonnegative_rank(
    rank: &Value,
    schema: &Map<String, Value>,
    clamped: bool,
) -> Result<(), String> {
    let Some(parts) = rank.as_array() else {
        return Ok(());
    };
    match parts.first().and_then(Value::as_str) {
        Some("Attribute") if parts.len() == 2 => {
            let field = parts[1].as_str().unwrap_or_default();
            let signed = schema
                .get(field)
                .is_some_and(|definition| matches!(field_type(definition), "int" | "float"));
            if signed && !clamped {
                return Err(format!(
                    "rank_by clauses must produce non-negative scores, but an attribute clause on signed numeric attribute '{field}' was used. Wrap this clause under a [\"Max\", [0, <clause>]] to fix this."
                ));
            }
            Ok(())
        }
        Some("Sum" | "Max") if parts.len() == 2 => {
            let children = parts[1].as_array().map(Vec::as_slice).unwrap_or_default();
            let floored = clamped || (parts[0] == "Max" && children.iter().any(Value::is_number));
            children
                .iter()
                .try_for_each(|child| validate_nonnegative_rank(child, schema, floored))
        }
        Some("Product") if parts.len() == 3 => {
            let expression = if parts[1].is_number() {
                &parts[2]
            } else {
                &parts[1]
            };
            validate_nonnegative_rank(expression, schema, clamped)
        }
        Some("Saturate" | "Decay") if parts.len() == 3 => {
            validate_nonnegative_rank(&parts[1], schema, true)
        }
        _ => Ok(()),
    }
}

/// Operators that make a clause a filter. Inside `rank_by` a filter scores 1 for a matching
/// row and 0 otherwise.
fn is_filter_operator(operator: &str) -> bool {
    matches!(
        operator,
        "Eq" | "NotEq"
            | "In"
            | "NotIn"
            | "Gt"
            | "Gte"
            | "Lt"
            | "Lte"
            | "Contains"
            | "NotContains"
            | "ContainsAny"
            | "NotContainsAny"
            | "AnyGt"
            | "AnyGte"
            | "AnyLt"
            | "AnyLte"
            | "ContainsAllTokens"
            | "ContainsAnyToken"
            | "ContainsTokenSequence"
            | "Glob"
            | "NotGlob"
            | "IGlob"
            | "NotIGlob"
            | "Regex"
            | "Fuzzy"
    )
}

fn is_ann(rank: &Value) -> bool {
    rank.as_array()
        .is_some_and(|a| a.len() == 3 && (a[1] == "ANN" || a[1] == "kNN"))
}

fn contains_knn(expression: &Value) -> bool {
    expression.as_array().is_some_and(|parts| {
        (parts.len() == 3 && parts[1] == "kNN") || parts.iter().any(contains_knn)
    })
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

pub(crate) fn score_rank(
    rank: &Value,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
    schema: &Map<String, Value>,
    metric: &str,
) -> f64 {
    let parts = rank.as_array().unwrap();
    if is_attribute_order(rank) {
        return 0.0;
    }
    if parts[0] == "Sum" || (parts[0] == "Max" && parts.len() == 2) {
        let scores = parts[1].as_array().unwrap().iter().map(|child| {
            child
                .as_f64()
                .unwrap_or_else(|| score_rank(child, row, corpus, schema, metric))
        });
        return if parts[0] == "Sum" {
            scores.sum()
        } else {
            scores.fold(f64::NEG_INFINITY, f64::max)
        };
    }
    if matches!(parts[0].as_str(), Some("Saturate" | "Decay")) {
        let value = score_rank(&parts[1], row, corpus, schema, metric).max(0.0);
        let midpoint = parts[2]["midpoint"]
            .as_f64()
            .or_else(|| parts[2]["midpoint"].as_str().and_then(parse_duration_ms))
            .unwrap();
        let exponent = parts[2]
            .get("exponent")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let value = value.powf(exponent);
        let midpoint = midpoint.powf(exponent);
        return if parts[0] == "Decay" {
            midpoint / (value + midpoint)
        } else {
            value / (value + midpoint)
        };
    }
    if parts[0] == "Dist" {
        let field = parts[1][1].as_str().unwrap();
        let Some(value) = row.get(field) else {
            return f64::INFINITY;
        };
        return if let (Some(value), Some(origin)) = (value.as_f64(), parts[2].as_f64()) {
            (value - origin).abs()
        } else if let (Some(value), Some(origin)) = (
            value.as_str().and_then(parse_datetime_ms),
            parts[2].as_str().and_then(parse_datetime_ms),
        ) {
            (value - origin).abs()
        } else {
            f64::INFINITY
        };
    }
    if parts[0] == "Product" {
        let (weight, expression) = if parts[1].is_number() {
            (&parts[1], &parts[2])
        } else {
            (&parts[2], &parts[1])
        };
        return weight.as_f64().unwrap() * score_rank(expression, row, corpus, schema, metric);
    }
    if parts[0] == "Attribute" {
        return row
            .get(parts[1].as_str().unwrap())
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
    }
    let field = parts[0].as_str().unwrap();
    match parts[1].as_str().unwrap() {
        "ANN" | "kNN" => {
            let query = parts[2].as_array().unwrap();
            let vector = row.get(field).and_then(Value::as_array);
            match vector {
                Some(vector) if query.first().is_some_and(Value::is_array) => query
                    .iter()
                    .map(|query_token| {
                        vector
                            .iter()
                            .filter_map(Value::as_array)
                            .map(|document_token| {
                                dense_distance(
                                    query_token.as_array().unwrap(),
                                    document_token,
                                    metric,
                                )
                            })
                            .fold(f64::INFINITY, f64::min)
                    })
                    .sum(),
                Some(vector) if vector.len() == query.len() => {
                    dense_distance(query, vector, metric)
                }
                _ => f64::INFINITY,
            }
        }
        "SparseKNN" => {
            let query = parts[2].as_object().unwrap();
            let Some(document) = row.get(field).and_then(Value::as_object) else {
                return 0.0;
            };
            let score: f32 = query
                .iter()
                .map(|(key, weight)| {
                    weight.as_f64().unwrap() as f32
                        * document.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32
                })
                .sum();
            serialized_f32(score)
        }
        "BM25" => bm25(
            field,
            &parts[2],
            parts
                .get(3)
                .and_then(|options| options.get("last_as_prefix"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            row,
            corpus,
            schema,
            false,
        ),
        operator if is_filter_operator(operator) => f64::from(filter_matches(rank, row, schema)),
        _ => 0.0,
    }
}

fn parse_datetime_ms(value: &str) -> Option<f64> {
    DateTime::<FixedOffset>::parse_from_rfc3339(value)
        .ok()
        .map(|time| time.timestamp_millis() as f64)
        .or_else(|| {
            NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .ok()
                .and_then(|date| date.and_hms_opt(0, 0, 0))
                .map(|time| time.and_utc().timestamp_millis() as f64)
        })
}

/// Whether `expression` is `["Dist", ["Attribute", field], origin]` over a datetime field,
/// the only input for which a duration string midpoint is meaningful.
fn is_datetime_dist(expression: &Value, schema: &Map<String, Value>) -> bool {
    expression.as_array().is_some_and(|parts| {
        parts.len() == 3
            && parts[0] == "Dist"
            && parts[1][1]
                .as_str()
                .and_then(|field| schema.get(field))
                .is_some_and(|definition| field_type(definition) == "datetime")
    })
}

fn parse_duration_ms(value: &str) -> Option<f64> {
    let split = value.find(|character: char| character.is_ascii_alphabetic())?;
    let number = value[..split].parse::<f64>().ok()?;
    if !number.is_finite() || number <= 0.0 {
        return None;
    }
    let factor = match &value[split..] {
        "ms" => 1.0,
        "s" => 1_000.0,
        "m" => 60_000.0,
        "h" => 3_600_000.0,
        "d" => 86_400_000.0,
        "w" => 604_800_000.0,
        _ => return None,
    };
    Some(number * factor)
}

fn dense_distance(query: &[Value], vector: &[Value], metric: &str) -> f64 {
    if query.len() != vector.len() {
        return f64::INFINITY;
    }
    if metric == "euclidean_squared" {
        let distance: f32 = vector
            .iter()
            .zip(query)
            .map(|(a, b)| (a.as_f64().unwrap() as f32 - b.as_f64().unwrap() as f32).powi(2))
            .sum();
        return serialized_f32(distance);
    }
    let dot: f32 = vector
        .iter()
        .zip(query)
        .map(|(a, b)| a.as_f64().unwrap() as f32 * b.as_f64().unwrap() as f32)
        .sum();
    let norm_a = vector
        .iter()
        .map(|a| (a.as_f64().unwrap() as f32).powi(2))
        .sum::<f32>()
        .sqrt();
    let norm_b = query
        .iter()
        .map(|a| (a.as_f64().unwrap() as f32).powi(2))
        .sum::<f32>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        f64::INFINITY
    } else {
        serialized_f32(1.0 - dot / (norm_a * norm_b))
    }
}

/// JSON numbers are f64; use the shortest decimal that round-trips to the
/// float32 score returned by the live service.
fn serialized_f32(value: f32) -> f64 {
    value.to_string().parse().unwrap()
}

fn serialized_score(score: f64) -> f64 {
    let narrow = score as f32;
    if narrow.is_finite() {
        serialized_f32(narrow)
    } else {
        score
    }
}

/// Corpus statistics of one full-text attribute, shared by every row a query scores.
struct FieldStats {
    average_length: f64,
    document_frequency: HashMap<String, usize>,
}

thread_local! {
    static FIELD_STATS: std::cell::RefCell<HashMap<String, std::rc::Rc<FieldStats>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Fragment ranking uses a different corpus from the outer query. Keep its BM25 statistics
/// separate so explicit `rank_fragments_by` expressions cannot affect sibling scores.
pub(crate) fn with_isolated_field_stats<T>(work: impl FnOnce() -> T) -> T {
    let saved = FIELD_STATS.with(|stats| std::mem::take(&mut *stats.borrow_mut()));
    let result = work();
    FIELD_STATS.with(|stats| *stats.borrow_mut() = saved);
    result
}

/// Drops per-request text caches. Every request that reads rows calls this first.
pub(crate) fn reset_text_caches() {
    FIELD_STATS.with(|stats| stats.borrow_mut().clear());
    crate::text::reset_cache();
}

fn field_stats(
    field: &str,
    analysis: &TextAnalysis,
    corpus: &BTreeMap<String, Map<String, Value>>,
) -> std::rc::Rc<FieldStats> {
    if let Some(stats) = FIELD_STATS.with(|stats| stats.borrow().get(field).cloned()) {
        return stats;
    }
    let mut lengths = Vec::new();
    let mut document_frequency = HashMap::new();
    for row in corpus.values() {
        let Some(value) = row.get(field) else {
            continue;
        };
        let tokens = analysis.analyze(value);
        lengths.push(tokens.len());
        let mut seen = std::collections::HashSet::new();
        for token in tokens.iter() {
            if seen.insert(token.text.as_str()) {
                *document_frequency.entry(token.text.clone()).or_insert(0) += 1;
            }
        }
    }
    let stats = std::rc::Rc::new(FieldStats {
        average_length: (lengths.iter().sum::<usize>() as f64 / lengths.len().max(1) as f64)
            .max(1.0),
        document_frequency,
    });
    FIELD_STATS.with(|cache| cache.borrow_mut().insert(field.to_owned(), stats.clone()));
    stats
}

fn bm25(
    field: &str,
    query: &Value,
    last_as_prefix: bool,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
    schema: &Map<String, Value>,
    computed: bool,
) -> f64 {
    let config = schema[field].get("full_text_search");
    let k1 = config
        .and_then(|value| value.get("k1"))
        .and_then(Value::as_f64)
        .unwrap_or(1.2) as f32;
    let b = config
        .and_then(|value| value.get("b"))
        .and_then(Value::as_f64)
        .unwrap_or(0.75) as f32;
    let k3 = config
        .and_then(|value| value.get("k3"))
        .and_then(Value::as_f64)
        .unwrap_or(8.0) as f32;
    let Some(analysis) = TextAnalysis::for_field(&schema[field]) else {
        return 0.0;
    };
    let doc_tokens = row
        .get(field)
        .map(|value| analysis.analyze(value))
        .unwrap_or_default();
    let mut query_tokens = analysis.query_tokens(query);
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let prefix = if last_as_prefix {
        query_tokens.pop()
    } else {
        None
    };
    // Live computed BM25 attributes use a fixed one-token average length and
    // ln(2) IDF. Rank clauses use the namespace's actual corpus statistics.
    let stats = (!computed).then(|| field_stats(field, &analysis, corpus));
    let avg_len = stats.as_ref().map_or(1.0, |stats| stats.average_length) as f32;
    let mut terms = HashMap::new();
    for token in query_tokens {
        *terms.entry(token).or_insert(0_usize) += 1;
    }
    let score: f32 = terms
        .into_iter()
        .map(|(term, query_freq)| {
            let freq = doc_tokens.iter().filter(|token| token.text == term).count() as f32;
            if freq == 0.0 {
                return 0.0;
            }
            let idf = if let Some(stats) = &stats {
                let df = stats.document_frequency.get(&term).copied().unwrap_or(0) as f32;
                let n = corpus.len() as f32;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            } else {
                std::f32::consts::LN_2
            };
            let norm = k1 * (1.0 - b + b * doc_tokens.len() as f32 / avg_len);
            let query_weight = query_freq as f32 * (k3 + 1.0) / (query_freq as f32 + k3);
            let term_weight = freq / (freq + norm) * (k1 + 1.0);
            query_weight * (idf * term_weight)
        })
        .sum();
    let score = score
        + f32::from(prefix.is_some_and(|prefix| {
            doc_tokens
                .iter()
                .any(|token| token.text.starts_with(&prefix))
        }));
    serialized_f32(score)
}

/// The live service's name for a JSON value's type in error messages.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
