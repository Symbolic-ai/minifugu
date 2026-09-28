use crate::store::{field_type, known_field, Namespace};
use chrono::{DateTime, FixedOffset};
use globset::GlobBuilder;
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

impl Namespace {
    pub fn query(&self, body: &Value) -> Result<Value, String> {
        let object = body.as_object().ok_or("query body must be an object")?;
        validate_query_options(object)?;
        if let Some(queries) = object.get("queries") {
            let queries = queries.as_array().ok_or("queries must be an array")?;
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
                        .then_with(|| a.1["id"].to_string().cmp(&b.1["id"].to_string()))
                });
                let rows = rows
                    .into_iter()
                    .skip(offset)
                    .take(limit)
                    .map(|(score, mut row)| {
                        row["$dist"] = json!(score);
                        row
                    })
                    .collect::<Vec<_>>();
                return Ok(
                    json!({"results":[{"rows":rows}],"billing":{},"performance":{"server_total_ms":0}}),
                );
            }
            if object
                .keys()
                .any(|key| !matches!(key.as_str(), "queries" | "consistency"))
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
            ) {
                return Err(format!("unsupported query field {key}"));
            }
        }
        if let Some(rank) = object.get("rank_by") {
            validate_rank(rank, &self.schema)?;
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
                .ok_or("compute_attributes must be an object")?;
            if computed.len() > 256 {
                return Err("compute_attributes exceeds 256 fields".into());
            }
            for (name, expression) in computed {
                if name == "id" || name == "$dist" || known_field(&self.schema, name) {
                    return Err(format!(
                        "computed attribute {name} conflicts with an existing field"
                    ));
                }
                validate_computed(expression, &self.schema)?;
            }
        }
        validate_query_options(object)?;
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
                if let Some(computed) = object.get("compute_attributes").and_then(Value::as_object)
                {
                    for (name, expression) in computed {
                        let value = computed_value(expression, row, &self.rows);
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

fn validate_query_options(object: &Map<String, Value>) -> Result<(), String> {
    if let Some(consistency) = object.get("consistency") {
        let consistency = consistency
            .as_object()
            .ok_or("consistency must be an object")?;
        if consistency.len() != 1
            || !matches!(
                consistency.get("level").and_then(Value::as_str),
                Some("strong" | "eventual")
            )
        {
            return Err("consistency.level must be strong or eventual".into());
        }
    }
    if object
        .get("distance_metric")
        .is_some_and(|value| value != "cosine_distance")
    {
        return Err("only cosine_distance is supported".into());
    }
    Ok(())
}

fn validate_computed(expression: &Value, schema: &Map<String, Value>) -> Result<(), String> {
    let parts = expression
        .as_array()
        .ok_or("computed attribute expression must be an array")?;
    if parts.len() != 3 {
        return Err("computed attribute requires a three-part expression".into());
    }
    if parts[1] == "BM25" {
        return validate_rank(expression, schema);
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
) -> Value {
    let mut rank = expression.clone();
    if rank[1] == "VectorDist" {
        rank[1] = json!("ANN");
    }
    let score = score_rank(&rank, row, corpus);
    if score.is_finite() {
        json!(score)
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
            "queries" | "rerank_by" | "limit" | "offset" | "consistency"
        )
    }) {
        return Err("unsupported multi-query field".into());
    }
    let parts = object["rerank_by"]
        .as_array()
        .ok_or("rerank_by must be an array")?;
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
    Ok((rank_constant, weights, limit as usize, offset as usize))
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
    let definition = &schema[field];
    let full_text_search = definition.get("full_text_search") == Some(&Value::Bool(true));
    let filterable = definition
        .get("filterable")
        .and_then(Value::as_bool)
        .unwrap_or(!full_text_search);
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
        if !schema.get(field).is_some_and(|definition| {
            definition
                .get("full_text_search")
                .is_some_and(|value| value == true || value.is_object())
        }) {
            return Err(format!(
                "attribute {field} is not configured for full-text search"
            ));
        }
        if !parts[2].is_string()
            && !parts[2]
                .as_array()
                .is_some_and(|tokens| tokens.iter().all(Value::is_string))
        {
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
        if parts.len() != 3 || field_type(&schema[field]) != "string" {
            return Err(format!(
                "{op} requires a string attribute and a three-part filter"
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
    if parts.len() == 4 {
        return Err(format!("{op} does not take options"));
    }
    match op {
        "Eq" | "NotEq" | "Gt" | "Gte" | "Lt" | "Lte" | "Contains" | "NotContains" => Ok(()),
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
        "NotIn" => !right.as_array().unwrap().contains(left),
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
        "AnyGt" | "AnyGte" | "AnyLt" | "AnyLte" => left.as_array().is_some_and(|values| {
            values.iter().any(|value| match parts[1].as_str().unwrap() {
                "AnyGt" => compare_values(value, right).is_some_and(|order| order > 0),
                "AnyGte" => compare_values(value, right).is_some_and(|order| order >= 0),
                "AnyLt" => compare_values(value, right).is_some_and(|order| order < 0),
                _ => compare_values(value, right).is_some_and(|order| order <= 0),
            })
        }),
        "ContainsAllTokens" | "ContainsAnyToken" | "ContainsTokenSequence" => {
            token_filter_matches(parts, left)
        }
        "Glob" | "NotGlob" | "IGlob" | "NotIGlob" => left.as_str().is_some_and(|text| {
            let matches = GlobBuilder::new(right.as_str().unwrap())
                .case_insensitive(matches!(parts[1].as_str(), Some("IGlob" | "NotIGlob")))
                .build()
                .unwrap()
                .compile_matcher()
                .is_match(text);
            if matches!(parts[1].as_str(), Some("NotGlob" | "NotIGlob")) {
                !matches
            } else {
                matches
            }
        }),
        "Regex" => left
            .as_str()
            .is_some_and(|text| Regex::new(right.as_str().unwrap()).unwrap().is_match(text)),
        _ => false,
    }
}

fn token_filter_matches(parts: &[Value], left: &Value) -> bool {
    let document = value_tokens(left);
    let terms = match &parts[2] {
        Value::String(query) => tokens(query),
        Value::Array(terms) => terms
            .iter()
            .flat_map(|term| tokens(term.as_str().unwrap()))
            .collect(),
        _ => return false,
    };
    if document.is_empty() || terms.is_empty() {
        return false;
    }
    let last_as_prefix = parts
        .get(3)
        .and_then(|options| options.get("last_as_prefix"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let matches_term = |index: usize, token: &str| {
        document.iter().any(|candidate| {
            if last_as_prefix && index + 1 == terms.len() {
                candidate.starts_with(token)
            } else {
                candidate == token
            }
        })
    };
    match parts[1].as_str().unwrap() {
        "ContainsAllTokens" => terms
            .iter()
            .enumerate()
            .all(|(index, token)| matches_term(index, token)),
        "ContainsAnyToken" => terms
            .iter()
            .enumerate()
            .any(|(index, token)| matches_term(index, token)),
        "ContainsTokenSequence" => document.windows(terms.len()).any(|window| window == terms),
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
    if parts.len() == 2 && matches!(parts[0].as_str(), Some("Sum" | "Max")) {
        let children = parts[1].as_array().ok_or("Sum/Max requires an array")?;
        if children.is_empty() {
            return Err("Sum/Max requires at least one expression".into());
        }
        for child in children {
            validate_rank(child, schema)?;
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
    if parts.len() == 3
        && parts[0].is_string()
        && matches!(
            parts[1].as_str(),
            Some("Eq" | "NotEq" | "In" | "NotIn" | "Gt" | "Gte" | "Lt" | "Lte")
        )
    {
        return validate_filter(rank, schema);
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
        "ANN" | "kNN"
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
            if operator == "ANN"
                && !field.starts_with("embed_")
                && schema[field].get("ann") != Some(&Value::Bool(true))
            {
                return Err(format!("attribute {field} does not enable ANN"));
            }
            Ok(())
        }
        "BM25" if parts.len() == 3 && parts[2].is_string() => {
            let definition = &schema[field];
            if !matches!(field_type(definition), "string" | "[]string")
                || !definition
                    .get("full_text_search")
                    .is_some_and(|value| value == true || value.is_object())
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

fn score_rank(
    rank: &Value,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
) -> f64 {
    let parts = rank.as_array().unwrap();
    if is_attribute_order(rank) {
        return 0.0;
    }
    if parts[0] == "Sum" || (parts[0] == "Max" && parts.len() == 2) {
        let scores = parts[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|child| score_rank(child, row, corpus));
        return if parts[0] == "Sum" {
            scores.sum()
        } else {
            scores.fold(f64::NEG_INFINITY, f64::max)
        };
    }
    if parts[0] == "Product" {
        let (weight, expression) = if parts[1].is_number() {
            (&parts[1], &parts[2])
        } else {
            (&parts[2], &parts[1])
        };
        return weight.as_f64().unwrap() * score_rank(expression, row, corpus);
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
        "Eq" | "NotEq" | "In" | "NotIn" | "Gt" | "Gte" | "Lt" | "Lte" => {
            f64::from(filter_matches(rank, row))
        }
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

fn value_tokens(value: &Value) -> Vec<String> {
    match value {
        Value::String(text) => tokens(text),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .flat_map(tokens)
            .collect(),
        _ => Vec::new(),
    }
}

fn bm25(
    field: &str,
    query: &str,
    row: &Map<String, Value>,
    corpus: &std::collections::BTreeMap<String, Map<String, Value>>,
) -> f64 {
    let doc_tokens = row.get(field).map(value_tokens).unwrap_or_default();
    let query_tokens = tokens(query);
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let lengths = corpus
        .values()
        .filter_map(|r| r.get(field).map(|value| value_tokens(value).len()))
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
                        .is_some_and(|value| value_tokens(value).contains(&term))
                })
                .count() as f64;
            let n = corpus.len() as f64;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let norm = 1.2 * (1.0 - 0.75 + 0.75 * doc_tokens.len() as f64 / avg_len);
            query_freq as f64 * idf * freq * 2.2 / (freq + norm)
        })
        .sum()
}
