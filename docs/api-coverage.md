# API coverage

MiniFugu targets small local fixtures. The table states both what works today and what still does not. It is not a claim of full Turbopuffer compatibility. The reference is Turbopuffer's [public OpenAPI description](https://github.com/turbopuffer/turbopuffer-openapi).

| Area | Supported now | Not yet supported |
| --- | --- | --- |
| `POST /v2/namespaces/{name}` | Row and column upserts/patches; ID and filter deletes; filter patches; per-row upsert/patch/delete conditions; affected IDs; schema declaration; cosine metric; local copy/branch | Cross-account copies, partial operations |
| Schema types | Scalar and scalar-array fields; fixed f16/f32 vectors; simple scalar inference; BM25 and token filters on string arrays | Sparse vectors, geospatial values, advanced schema types |
| `POST /v2/namespaces/{name}/query` | Single and multiquery; `rank_by`; `filters`; `top_k`; numeric and `{total,per}` limits; offset; include/exclude attributes; Count/Sum and simple grouped aggregations; RRF fusion with weights and rank constant; computed BM25 and vector distance fields; `strong`/`eventual` consistency options both read the current local snapshot | Other computed expressions, query explain, distinct consistency semantics, advanced group expressions, other rerank functions |
| Filters | `And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `NotIn`, `Gt`, `Gte`, `Lt`, `Lte`, array containment, `AnyGt`/`AnyGte`/`AnyLt`/`AnyLte`, `ContainsAllTokens`, `ContainsAnyToken`, `ContainsTokenSequence`, `last_as_prefix`, `Glob`/`NotGlob`/`IGlob`/`NotIGlob`, `Regex` | Fuzzy and geo filters; exact live tokenizer behavior |
| Ranking | Exact cosine ANN and filtered kNN; BM25; scalar and multi-attribute ordering; `Sum`/list `Max`/`Product`, numeric `Attribute`, filter predicates as scores, reciprocal rank fusion | Sparse KNN, `Saturate`/`Decay`/`Dist` and other ranking expressions. Exact ANN results and BM25 scores can differ from the live service |
| Namespace management | Paginated `GET /v1/namespaces` with prefix; get/update schema; get metadata; delete; local copy/branch | Pinning, read-only metadata mutation, sharding, cache warm, recall debug route |
| Persistence | Optional single JSON snapshot with sync and atomic rename | Multi-process locking, production-scale storage |
| Embeddings | Offline deterministic hash; opt-in OpenAI native embeddings | Other hosted embedding providers; offline hashes are not semantic |

API errors are JSON with `status` and `error`. Error text and some metadata fields are not byte-for-byte compatible. Unknown write and query fields fail with HTTP 400, so unsupported operations cannot silently pass a contract test. A listed feature can still differ on edge cases; the [compatibility tests](../tests/compatibility.rs) exercise synthetic rows against both MiniFugu and an optional live development account.

The live compatibility suite uses only disposable synthetic rows. It does not read application or customer records.
