# API coverage

MiniFugu implements **all 10 distinct HTTP method/path pairs** in Turbopuffer's [public OpenAPI description](https://github.com/turbopuffer/turbopuffer-openapi), checked on 2026-09-29. The specification describes single and multiquery on the same URL, so it has 11 operation entries. Route coverage is 100%; behavioral parity is not. MiniFugu is intended for small, synthetic local fixtures.

| Documented route | Supported behavior | Limits |
| --- | --- | --- |
| `GET /v1/namespaces` | Sorted namespace list, prefix, cursor, page size | Local namespaces only |
| `GET /v1/namespaces/{name}/schema` | Returns the local schema | Schema normalization differs from the service |
| `POST /v1/namespaces/{name}/schema` | Updates an existing schema | Some advanced schema options are unavailable |
| `GET /v1/namespaces/{name}/metadata` | Schema, row and byte estimates, durable timestamps, encryption and index fields | Byte counts are local estimates; index is always marked up to date; `encryption.sse` is a compatibility value and local snapshots are not encrypted |
| `GET /v1/namespaces/{name}/hint_cache_warm` | Returns HTTP 202 and the documented acceptance body | Local scans need no cache warming |
| `POST /v1/namespaces/{name}/_debug/recall` | Runs exact vector searches with optional filters and ground truth | Recall is 1.0 for exact local search; sampling and index diagnostics differ |
| `POST /v2/namespaces/{name}` | Row/column upsert and patch, ID/filter deletes, conditions, affected IDs, schema, local copy/branch; partial filter flags complete all matching small local rows; backpressure flag accepted for upserts and ID deletes | Cross-account copy, service-style partial chunking, sharding and encryption configuration unavailable |
| `DELETE /v2/namespaces/{name}` | Deletes rows and schema, including durable data | Local persistence only |
| `POST /v2/namespaces/{name}/query` | Single/multiquery, filters, dense/sparse/multi-vector and BM25 ranking, numeric scoring, RRF, computed scores, basic aggregation, projection and limits | Highlight, some advanced text options and server metrics unavailable |
| `POST /v2/namespaces/{name}/explain_query` | Validates and describes the local exact scan plan | Plan text describes MiniFugu, not Turbopuffer's distributed planner |

| Area | Supported now | Not yet supported or behavior differs |
| --- | --- | --- |
| Schema and data | Scalar and scalar-array fields; `bytes` as base64; fixed f16/f32/i8 vectors; `{}f16` sparse vectors; `[][N]f32` multi-vectors; float arrays or float32 base64 dense input; native-width f16/f32/i8 base64 dense output; simple scalar inference; BM25 on string arrays; object-form FTS tuning for `k1`, `b`, `k3` | FTS tokenizer/language/stemming/stopwords/ascii folding/max-token-length settings and other advanced schema options |
| Filters | `And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `NotIn`, `Gt`, `Gte`, `Lt`, `Lte`, array containment, `AnyGt`/`AnyGte`/`AnyLt`/`AnyLte`, token containment/sequence, glob, regex and fuzzy substring matching with edit-distance thresholds | Exact live tokenizer and index behavior; advanced regex dialect differences |
| Ranking | Exact cosine or squared Euclidean ANN/kNN, sparse dot-product KNN, late-interaction multi-vector ranking, BM25 with optional last-token prefix, scalar and multi-attribute ordering, `Sum`/`Max`/`Product`, numeric `Attribute`, `Saturate`/`Decay`/`Dist`, filter predicates as scores, weighted RRF | Approximate ANN, live BM25 score precision, other ranking expressions |
| Query details | Integer `top_k`; integer or `{total,per}` limit; offset; include/exclude attributes; Count/Sum and simple grouped aggregates; computed BM25/vector distance; `strong`/`eventual` options both read the current local snapshot; local billing/performance estimates | Text highlighting, advanced grouping, distinct consistency semantics, exact cloud billing/performance values |
| Persistence | Optional single JSON snapshot with sync and atomic rename; metadata timestamps persist | Multi-process locking, production-scale storage |
| Embeddings | Offline deterministic hash, opt-in OpenAI native embeddings | Other hosted embedding providers; offline hashes are not semantic |

Sparse KNN, fuzzy filtering, byte fields and numeric ranking are exercised against disposable synthetic namespaces in the optional [live compatibility test](../tests/compatibility.rs). Multi-vector behavior is covered locally; the available development account rejects vector-array types as a gated feature, so live parity for that feature is unverified.

API errors are JSON with `status` and `error`. Unknown write and query fields fail with HTTP 400, so unsupported operations cannot silently pass a contract test. Error text, some metadata and response fields, and edge cases can differ. Compatibility tests never read application or customer records.
