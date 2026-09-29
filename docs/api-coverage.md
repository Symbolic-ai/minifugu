# API coverage

MiniFugu implements **all 10 distinct HTTP method/path pairs** in Turbopuffer's [public OpenAPI description](https://github.com/turbopuffer/turbopuffer-openapi), checked on 2026-09-29. The specification describes single and multiquery on the same URL, so it has 11 operation entries. Route coverage is 100%; behavioral parity is not. MiniFugu is intended for small, synthetic local fixtures.

| Documented route | Supported behavior | Limits |
| --- | --- | --- |
| `GET /v1/namespaces` | Sorted namespace list, prefix, cursor, page size | Local namespaces only |
| `GET /v1/namespaces/{name}/schema` | Returns the live service's normalized schema shape for supported attributes | Unsupported schema features remain unavailable |
| `POST /v1/namespaces/{name}/schema` | Updates an existing schema, including supported text analysis options | Some advanced schema options are unavailable |
| `GET /v1/namespaces/{name}/metadata` | Schema, row and byte estimates, durable timestamps, encryption and index fields | Byte counts are local estimates; index is always marked up to date; `encryption.sse` is a compatibility value and local snapshots are not encrypted |
| `GET /v1/namespaces/{name}/hint_cache_warm` | Returns HTTP 202 and the documented acceptance body | Local scans need no cache warming |
| `POST /v1/namespaces/{name}/_debug/recall` | Runs exact vector searches with optional filters and ground truth | Recall is 1.0 for exact local search; sampling and index diagnostics differ |
| `POST /v2/namespaces/{name}` | Row/column upsert and patch, ID/filter deletes, conditions, affected IDs, schema, local copy/branch; partial filter flags complete all matching small local rows; backpressure flag accepted for upserts and ID deletes | Cross-account copy, service-style partial chunking, sharding and encryption configuration unavailable |
| `DELETE /v2/namespaces/{name}` | Deletes rows and schema, including durable data | Local persistence only |
| `POST /v2/namespaces/{name}/query` | Single/multiquery, filters, dense/sparse/multi-vector and BM25 ranking, numeric scoring, RRF, computed scores, text highlighting, Count/Sum aggregation with multi-attribute grouping, projection and limits | Server metrics and some advanced expressions unavailable |
| `POST /v2/namespaces/{name}/explain_query` | Validates and describes the local exact scan plan | Plan text describes MiniFugu, not Turbopuffer's distributed planner |

| Area | Supported now | Not yet supported or behavior differs |
| --- | --- | --- |
| Schema and data | Requests up to the documented 512 MB upsert limit; 8 MiB attribute values, which also bounds a multi-vector by its total float32 size; scalar and scalar-array fields; `bytes` as base64; fixed f16/f32/i8 vectors; `{}f16` sparse vectors; `[][N]f32` multi-vectors; float arrays or float32 base64 dense input; native-width f16/f32/i8 base64 dense output; scalar inference; BM25 on string arrays; object-form FTS tuning for `k1`, `b`, `k3`, tokenizer, language, stemming, stopwords, case sensitivity, ASCII folding and maximum token length | Some advanced schema options |
| Filters | `And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `NotIn`, `Gt`, `Gte`, `Lt`, `Lte`, array containment, `AnyGt`/`AnyGte`/`AnyLt`/`AnyLte`, token containment/sequence, glob, regex and fuzzy substring matching with edit-distance thresholds. `full_text_search`, `regex`, `glob` and `fuzzy` make `filterable=false` the default, as on the live service | Advanced regex dialect and index behavior differences |
| Ranking | Exact cosine or squared Euclidean ANN/kNN, sparse dot-product KNN, late-interaction multi-vector ranking, BM25 with optional last-token prefix, scalar and multi-attribute ordering, `Sum`/`Max`/`Product`, scalar floors inside `Max` (`["Max",[0, clause]]`), numeric `Attribute` with the live rule that signed attributes need a floor, `Saturate`/`Decay`/`Dist` with duration midpoints in `ms`/`s`/`m`/`h`/`d`/`w`, filter predicates as scores, weighted RRF. Attribute-derived scores return every row, including zero scores; text, filter and sparse clauses return only matching rows. Equal scores use numeric ID order for integer IDs | Approximate ANN, live BM25 score precision, other ranking expressions |
| Query details | Integer `top_k`; integer or `{total,per}` limit; offset; include/exclude attributes; Count/Sum with multiple group fields, null groups and stable float sums; computed BM25/vector distance and text highlights; `strong`/`eventual` options both read the current local snapshot; local billing/performance estimates | Distinct consistency semantics, exact cloud billing/performance values |
| Persistence | Optional single JSON snapshot with sync and atomic rename; metadata timestamps persist | Multi-process locking, production-scale storage |
| Embeddings | Offline deterministic hash, opt-in OpenAI native embeddings | Other hosted embedding providers; offline hashes are not semantic |

## Behavioral parity checks

These checks compare MiniFugu with **disposable, generated namespaces** on live Turbopuffer. They cover the listed inputs, not every possible value or option.

| Area | Compared behavior | Evidence |
| --- | --- | --- |
| Text analysis | BM25 matches across 13 analyzer configurations and 54 query strings, including tokenizer versions, stemming, stopwords, French, case sensitivity, ASCII folding and token length | [Captured live expectations](../tests/ranking.rs) in `text_analysis_matches_live_tokenization` |
| Highlighting | Returned fragments and offsets, ordering, limits and invalid options across 21 live cases | [Highlight fixture](../tests/fixtures/live_highlight.json) replayed by `highlights_match_the_live_service` |
| Schema responses | Normalized `GET /schema` and metadata schema entries for 23 attributes, including inferred fields and full-text defaults | [Schema fixture](../tests/fixtures/live_schema.json) replayed by `schema_views_match_the_live_service` |
| Aggregations | `Count` and `Sum`, multiple group fields, null groups, empty `group_by`, float sum output, and rejected `id` or duplicate group fields | [Aggregation edge test](../tests/http.rs) and [live differential test](../tests/differential.rs) |
| Query responses | Status, row IDs and order, `$dist` scores, and aggregation values for 36 generated queries over four 12-row namespaces; numeric scores use a relative tolerance of `1e-5` with an absolute floor of `1e-5` | [Seeded differential test](../tests/differential.rs) |
| Errors | JSON error body; HTTP 400 for malformed JSON and common semantic errors; HTTP 422 for tested request-shape errors | [HTTP response tests](../tests/http.rs) and live probes |

For example, an empty `group_by: []` returns `aggregations`, just like an omitted `group_by`. A float `Sum` of zero stays `0.0`. Equal BM25 scores on integer IDs such as `5` and `11` return ID `5` first. These cases have local regression tests and were checked against live responses.

Sparse KNN, fuzzy filtering, byte fields, numeric ranking, `Max` floors and default filterability are exercised against disposable synthetic namespaces in the optional [live compatibility test](../tests/compatibility.rs). Multi-vector behavior is covered locally; the available development account rejects vector-array types as a gated feature, so live parity for that feature is unverified.

The optional [seeded differential test](../tests/differential.rs) runs only when `TURBOPUFFER_BASE_URL` and `TURBOPUFFER_API_KEY` are set. Run `cargo test --locked --test differential` to compare both services. It uses only generated rows and deletes its namespaces when finished. The normal suite replays the captured live expectations without an account. The differential comparison excludes billing, performance, and error text, which are service-specific or still differ.

API errors are JSON with `status` and `error`. Malformed JSON returns HTTP 400; recognized request shape errors return HTTP 422; semantic validation errors generally return HTTP 400. MiniFugu deliberately rejects unknown write and query fields instead of silently accepting unsupported operations, which differs from some live endpoints. Error text, some metadata and response fields, and edge cases can differ. Compatibility tests never read application or customer records.
