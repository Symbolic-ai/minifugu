# API coverage

MiniFugu targets small local fixtures. This table describes implemented behavior, not a claim of full Turbopuffer compatibility. The reference is Turbopuffer's [public OpenAPI description](https://github.com/turbopuffer/turbopuffer-openapi).

| Area | Implemented | Current limit |
| --- | --- | --- |
| v2 writes | `upsert_rows`, `patch_rows`, `deletes`, `delete_by_filter`, `patch_by_filter`, `return_affected_ids`, schema declaration, cosine metric | Column writes, conditional writes, copy/branch, and partial operations return 400 |
| Schemas | Scalar fields and fixed f16 vectors; simple scalar inference; unknown query fields rejected | No array, geospatial, or advanced schema types |
| Queries | Single and multiquery; `rank_by`; `filters`; `top_k`, numeric or rows `limit`; `offset`; include/exclude attributes | No aggregations, server-side reranking, computed attributes, query explain, or consistency modes |
| Filters | `And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `Gte`, `Lte` | No text token operators, prefix, or geo filters |
| Ranking | Exact cosine ANN, BM25, numeric ascending/descending, BM25 `Sum`/`Product` | Exact results differ from approximate ANN; BM25 tokenization and scores may differ |
| Namespace management | List, get/update schema, get metadata, delete | No pinning, read-only metadata mutation, sharding, cache warm, or recall debug route |
| Persistence | Optional single JSON snapshot with sync and atomic rename | One process at a time; no multi-process locking or production-scale storage |
| Embeddings | Offline deterministic hash; opt-in OpenAI native embeddings | Offline hashes are not semantic; model output depends on external provider when enabled |

API errors are JSON with `status` and `error`. Error text and some metadata fields are intentionally not byte-for-byte compatible. Unknown write and query fields fail with HTTP 400, so unsupported operations cannot silently pass a contract test.

The live compatibility suite uses only disposable synthetic rows. It does not read application or customer records.
