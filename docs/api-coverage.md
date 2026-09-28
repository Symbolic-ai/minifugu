# API coverage

MiniFugu targets small local fixtures. This table describes implemented behavior, not a claim of full Turbopuffer compatibility. The reference is Turbopuffer's [public OpenAPI description](https://github.com/turbopuffer/turbopuffer-openapi).

| Area | Implemented | Current limit |
| --- | --- | --- |
| v2 writes | Row and column upserts/patches, ID deletes, filter patch/delete, affected IDs, schema declaration, cosine metric, local namespace copy/branch | Conditional writes, cross-account copies, and partial operations return 400 |
| Schemas | Scalar and scalar-array fields, fixed f16/f32 vectors; simple scalar inference; unknown query fields rejected | No sparse vectors, geospatial, or advanced schema types |
| Queries | Single and multiquery; `rank_by`; `filters`; `top_k`, numeric or `{total,per}` `limit`; `offset`; include/exclude attributes; Count/Sum and simple grouped aggregations | No server-side reranking, computed attributes, query explain, or consistency modes; advanced group expressions are unsupported |
| Filters | `And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `Gt`, `Gte`, `Lt`, `Lte`, array containment | No text token operators, prefix, or geo filters |
| Ranking | Exact cosine ANN, BM25, scalar ascending/descending and multiple attribute order clauses, BM25 `Sum`/`Product` | Exact results differ from approximate ANN; BM25 tokenization and scores may differ |
| Namespace management | Paginated list with prefix, get/update schema, get metadata, delete, local copy/branch | No pinning, read-only metadata mutation, sharding, cache warm, or recall debug route |
| Persistence | Optional single JSON snapshot with sync and atomic rename | One process at a time; no multi-process locking or production-scale storage |
| Embeddings | Offline deterministic hash; opt-in OpenAI native embeddings | Offline hashes are not semantic; model output depends on external provider when enabled |

API errors are JSON with `status` and `error`. Error text and some metadata fields are intentionally not byte-for-byte compatible. Unknown write and query fields fail with HTTP 400, so unsupported operations cannot silently pass a contract test.

The live compatibility suite uses only disposable synthetic rows. It does not read application or customer records.
