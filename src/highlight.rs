//! The `Highlight` computed attribute: splits a full-text attribute into fragments, scores
//! them with BM25 using the row's own fragments as the corpus, and returns the best ones.

use crate::query::{score_rank, validate_rank, with_isolated_field_stats};
use crate::text::{self, TextAnalysis};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

/// The attribute name a fragment expression uses for the fragment text.
const FRAGMENT: &str = "$fragment";
const DEFAULT_FRAGMENT_LIMIT: u64 = 3;
const MAX_FRAGMENT_LIMIT: u64 = 1024;

pub(crate) fn is_highlight(expression: &Value) -> bool {
    expression.get(0) == Some(&json!("Highlight"))
}

/// Validates `["Highlight", field, options?]` with the live service's messages.
pub(crate) fn validate(
    name: &str,
    expression: &Value,
    rank_by: Option<&Value>,
    schema: &Map<String, Value>,
) -> Result<(), String> {
    let parts = expression.as_array().unwrap();
    if !(2..=3).contains(&parts.len()) {
        return Err(crate::shape_error(
            "ComputeAttributeInput: Highlight takes an attribute and optional options",
        ));
    }
    let field = parts[1]
        .as_str()
        .ok_or_else(|| crate::shape_error("Highlight attribute must be a string"))?;
    let definition = schema
        .get(field)
        .ok_or_else(|| format!("attribute \"{field}\" does not exist in namespace"))?;
    let Some(analysis) = TextAnalysis::for_field(definition) else {
        return Err(format!(
            "computed attribute `{name}` references `{field}`, which is not enabled for full-text search; this requires a schema update"
        ));
    };
    if analysis.is_pre_tokenized() {
        return Err(format!(
            "computed attribute `{name}` references `{field}`, which uses a pre-tokenized tokenizer that does not support highlighting"
        ));
    }
    let options = options(expression)?;
    if let Some(unit) = options.get("fragment_by") {
        if !matches!(
            unit.as_str(),
            Some("none" | "sentence" | "paragraph" | "word")
        ) {
            return Err(crate::shape_error(format!(
                "ComputeAttributeInput::HighlightWithConfig: unknown variant `{}`, expected one of `none`, `sentence`, `paragraph`, `word`",
                unit.as_str().unwrap_or_default()
            )));
        }
    }
    if let Some(unit) = options.get("include_offsets") {
        if !matches!(unit.as_str(), Some("utf-8" | "utf-16" | "codepoints")) {
            return Err(crate::shape_error(
                "ComputeAttributeInput::HighlightWithConfig: unknown variant for `include_offsets`, expected one of `utf-8`, `utf-16`, `codepoints`",
            ));
        }
    }
    if let Some(limit) = options.get("fragment_limit") {
        let limit = limit
            .as_u64()
            .ok_or_else(|| crate::shape_error("`fragment_limit` must be an integer"))?;
        if !(1..=MAX_FRAGMENT_LIMIT).contains(&limit) {
            return Err(format!(
                "computed attribute `{name}`: `fragment_limit` must be between 1 and {MAX_FRAGMENT_LIMIT}"
            ));
        }
    }
    let expression = fragment_expression(&options, rank_by, field).ok_or_else(|| {
        format!(
            "computed attribute `{name}`: cannot infer `rank_fragments_by` because the query `rank_by` has no BM25 clause over `{field}`; provide `rank_fragments_by` explicitly"
        )
    })?;
    let mut fragment_schema = schema.clone();
    fragment_schema.insert(FRAGMENT.into(), definition.clone());
    validate_rank(&expression, &fragment_schema)
}

fn options(expression: &Value) -> Result<Map<String, Value>, String> {
    let Some(options) = expression.get(2) else {
        return Ok(Map::new());
    };
    let options = options
        .as_object()
        .ok_or_else(|| crate::shape_error("Highlight options must be an object"))?;
    if let Some(key) = options.keys().find(|key| {
        !matches!(
            key.as_str(),
            "rank_fragments_by" | "fragment_by" | "fragment_limit" | "include_offsets"
        )
    }) {
        return Err(crate::shape_error(format!(
            "ComputeAttributeInput::HighlightWithConfig: unknown field `{key}`, expected one of `rank_fragments_by`, `fragment_by`, `fragment_limit`, `include_offsets`"
        )));
    }
    Ok(options.clone())
}

/// The expression that ranks fragments: `rank_fragments_by`, or else the query's BM25
/// clauses over the highlighted attribute, rewritten to read the fragment.
fn fragment_expression(
    options: &Map<String, Value>,
    rank_by: Option<&Value>,
    field: &str,
) -> Option<Value> {
    if let Some(expression) = options.get("rank_fragments_by") {
        return Some(expression.clone());
    }
    let mut clauses = Vec::new();
    collect_bm25(rank_by?, field, &mut clauses);
    match clauses.len() {
        0 => None,
        1 => clauses.pop(),
        _ => Some(json!(["Sum", clauses])),
    }
}

fn collect_bm25(expression: &Value, field: &str, clauses: &mut Vec<Value>) {
    let Some(parts) = expression.as_array() else {
        return;
    };
    if parts.len() >= 3 && parts[0] == field && parts[1] == "BM25" {
        let mut clause = parts.clone();
        clause[0] = json!(FRAGMENT);
        clauses.push(Value::Array(clause));
        return;
    }
    for part in parts {
        collect_bm25(part, field, clauses);
    }
}

/// Query tokens of every fragment BM25 clause, used to mark matches.
fn query_terms(expression: &Value, analysis: &TextAnalysis, terms: &mut HashSet<String>) {
    let Some(parts) = expression.as_array() else {
        return;
    };
    if parts.len() >= 3 && parts[0] == FRAGMENT && parts[1] == "BM25" {
        terms.extend(analysis.query_tokens(&parts[2]));
        return;
    }
    for part in parts {
        query_terms(part, analysis, terms);
    }
}

struct Fragment {
    input_index: usize,
    range: Range<usize>,
    score: f64,
}

/// Computes the highlight of one row. Validation guarantees the options are well formed.
pub(crate) fn compute(
    expression: &Value,
    rank_by: Option<&Value>,
    row: &Map<String, Value>,
    schema: &Map<String, Value>,
    metric: &str,
) -> Value {
    let field = expression[1].as_str().unwrap();
    let definition = &schema[field];
    let analysis = TextAnalysis::for_field(definition).unwrap();
    let options = options(expression).unwrap_or_default();
    let Some(ranking) = fragment_expression(&options, rank_by, field) else {
        return json!([]);
    };
    let inputs: Vec<&str> = match row.get(field) {
        Some(Value::String(text)) => vec![text.as_str()],
        Some(Value::Array(values)) => values.iter().filter_map(Value::as_str).collect(),
        _ => return json!([]),
    };
    let unit = options
        .get("fragment_by")
        .and_then(Value::as_str)
        .unwrap_or("sentence");
    let mut fragments = Vec::new();
    for (input_index, input) in inputs.iter().enumerate() {
        let ranges = match unit {
            "none" => std::iter::once(0..input.len()).collect(),
            "paragraph" => text::paragraphs(input),
            "word" => analysis
                .analyze(&json!(input))
                .iter()
                .map(|token| token.byte_range.clone())
                .collect(),
            _ => text::sentences(input),
        };
        fragments.extend(ranges.into_iter().map(|range| Fragment {
            input_index,
            range,
            score: 0.0,
        }));
    }

    // Score every fragment against a corpus made of this row's fragments.
    let mut fragment_schema = schema.clone();
    fragment_schema.insert(FRAGMENT.into(), definition.clone());
    let corpus: BTreeMap<String, Map<String, Value>> = fragments
        .iter()
        .enumerate()
        .map(|(index, fragment)| {
            // Auto-derived ranks read only $fragment. Explicit ranks may read other row fields.
            let mut fragment_row = if options.contains_key("rank_fragments_by") {
                row.clone()
            } else {
                Map::new()
            };
            fragment_row.insert(
                FRAGMENT.into(),
                json!(&inputs[fragment.input_index][fragment.range.clone()]),
            );
            (format!("{index:08}"), fragment_row)
        })
        .collect();
    with_isolated_field_stats(|| {
        for (fragment, fragment_row) in fragments.iter_mut().zip(corpus.values()) {
            fragment.score = score_rank(&ranking, fragment_row, &corpus, &fragment_schema, metric);
        }
    });

    fragments.retain(|fragment| fragment.score.is_finite() && fragment.score > 0.0);
    fragments.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.input_index.cmp(&b.input_index))
            .then(a.range.start.cmp(&b.range.start))
    });
    let limit = options
        .get("fragment_limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_FRAGMENT_LIMIT) as usize;
    fragments.truncate(limit);

    let offsets = options.get("include_offsets").and_then(Value::as_str);
    let offset_data = offsets.map(|_| {
        let mut terms = HashSet::new();
        query_terms(&ranking, &analysis, &mut terms);
        let tokens = inputs
            .iter()
            .map(|input| analysis.analyze(&json!(input)))
            .collect::<Vec<_>>();
        (terms, tokens)
    });
    let is_array = matches!(row.get(field), Some(Value::Array(_)));
    let highlights = fragments
        .iter()
        .map(|fragment| {
            let input = inputs[fragment.input_index];
            let mut highlight = Map::new();
            highlight.insert("text".into(), json!(&input[fragment.range.clone()]));
            if let Some(unit) = offsets {
                let position = |byte: usize| offset(input, byte, unit);
                highlight.insert(
                    "fragment_range".into(),
                    json!([position(fragment.range.start), position(fragment.range.end)]),
                );
                let (terms, tokens) = offset_data.as_ref().unwrap();
                let matches: Vec<Value> = tokens[fragment.input_index]
                    .iter()
                    .filter(|token| {
                        token.byte_range.start >= fragment.range.start
                            && token.byte_range.end <= fragment.range.end
                            && terms.contains(&token.text)
                    })
                    .map(|token| {
                        json!([
                            position(token.byte_range.start),
                            position(token.byte_range.end)
                        ])
                    })
                    .collect();
                highlight.insert("match_ranges".into(), Value::Array(matches));
                if is_array {
                    highlight.insert("array_index".into(), json!(fragment.input_index));
                }
            }
            Value::Object(highlight)
        })
        .collect();
    Value::Array(highlights)
}

/// Converts a byte offset in `text` to the requested unit.
fn offset(text: &str, byte: usize, unit: &str) -> usize {
    match unit {
        "utf-16" => text[..byte].encode_utf16().count(),
        "codepoints" => text[..byte].chars().count(),
        _ => byte,
    }
}
