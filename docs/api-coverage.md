# API coverage

MiniFugu implements **all 10 distinct HTTP method/path pairs** in Turbopuffer's [public OpenAPI description](https://github.com/turbopuffer/turbopuffer-openapi) as checked on 2026-09-28. The specification describes single and multiquery on the same URL, so it has 11 operation entries. Route coverage is 100%; request and behavior coverage is **not** 100%. This page lists both supported behavior and remaining gaps. MiniFugu is intended for small, synthetic local fixtures.

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
| `POST /v2/namespaces/{name}/query` | Single/multiquery, filters, vector/BM25 ranking, RRF, computed scores, basic aggregation, projection and limits | Advanced query expressions and server metrics unavailable |
| `POST /v2/namespaces/{name}/explain_query` | Validates and describes the local exact scan plan | Plan text describes MiniFugu, not Turbopuffer's distributed planner |

| Area | Supported now | Not yet supported or behavior differs |
| --- | --- | --- |
| Schema and data | Scalar and scalar-array fields; fixed f16/f32/i8 vectors; float arrays or float32 base64 input; native-width f16/f32/i8 base64 output; simple scalar inference; BM25 on string arrays | Sparse and multi-vectors, geospatial values, bytes, fuzzy indexing, object-form full-text configuration, advanced schema types and options |
| Filters | `And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `NotIn`, `Gt`, `Gte`, `Lt`, `Lte`, array containment, `AnyGt`/`AnyGte`/`AnyLt`/`AnyLte`, token containment/sequence, glob and regex | Fuzzy and geo filters; exact live tokenizer and index behavior |
| Ranking | Exact cosine or squared Euclidean ANN/kNN, BM25, scalar and multi-attribute ordering, `Sum`/`Max`/`Product`, numeric `Attribute`, filter predicates as scores, weighted RRF | Sparse KNN, `Saturate`/`Decay`/`Dist`, other ranking expressions; approximate ANN and live BM25 scores |
| Query details | Integer `top_k`; integer or `{total,per}` limit; offset; include/exclude attributes; Count/Sum and simple grouped aggregates; computed BM25/vector distance; `strong`/`eventual` options both read the current local snapshot; local billing/performance estimates | Legacy `{type:"rows",value:N}` limit; advanced grouping, other rerank functions, distinct consistency semantics, exact cloud billing/performance values |
| Persistence | Optional single JSON snapshot with sync and atomic rename; metadata timestamps persist | Multi-process locking, production-scale storage |
| Embeddings | Offline deterministic hash, opt-in OpenAI native embeddings | Other hosted embedding providers; offline hashes are not semantic |

API errors are JSON with `status` and `error`. Unknown write and query fields fail with HTTP 400, so unsupported operations cannot silently pass a contract test. Error text, some metadata and response fields, and edge cases can differ. The [compatibility tests](../tests/compatibility.rs) exercise disposable synthetic rows against MiniFugu and optionally a live development account; they never read application or customer records.
