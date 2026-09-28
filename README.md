# MiniFugu

MiniFugu is a small, MIT licensed Rust HTTP emulator for a useful subset of the [Turbopuffer](https://turbopuffer.com/docs/api-overview) v2 API. It is intended for local development and contract tests. It accepts any nonempty bearer token, so ordinary tests need no Turbopuffer account.

It stores rows in memory, validates query fields against namespace schemas, and runs actual exact cosine and BM25 ranking over small datasets. A missing field returns HTTP 400 even if a filter would otherwise match no rows. That catches schema mistakes hidden by mocked client responses.

## Start

Requires Rust 1.98 or newer.

```sh
cargo run
# listening on http://127.0.0.1:8787
```

Set `MINIFUGU_LISTEN=0.0.0.0:8787` to change the listen address. Each process starts with an empty store; stopping it removes all namespaces and rows.

```sh
curl -sS http://127.0.0.1:8787/v2/namespaces/demo \
  -H 'Authorization: Bearer dummy' -H 'Content-Type: application/json' \
  -d '{"schema":{"id":"uint","title":{"type":"string","full_text_search":true},"vector":{"type":"[2]f16","ann":true}},"distance_metric":"cosine_distance","upsert_rows":[{"id":1,"title":"small fugu","vector":[1,0]},{"id":2,"title":"blue whale","vector":[0,1]}]}'

curl -sS http://127.0.0.1:8787/v2/namespaces/demo/query \
  -H 'Authorization: Bearer dummy' -H 'Content-Type: application/json' \
  -d '{"queries":[{"rank_by":["vector","ANN",[1,0]],"limit":2},{"rank_by":["title","BM25","fugu"],"limit":2}]}'
```

The base URL is `http://127.0.0.1:8787`. Point your client at it and use any dummy token. For example, configure a client whose default URL is `https://aws-us-west-2.turbopuffer.com` to use MiniFugu's base URL in tests.

## Embeddings

Rows with explicit vector attributes are ranked by exact cosine distance. For schemas with a native `embed` field, MiniFugu creates an `embed_<field>` vector when writing text. For example:

```json
{"content":{"type":"string","full_text_search":true,"embed":{"model":"openai/text-embedding-3-small","dims":1536}}}
```

The default embedding mode uses deterministic token hashing. It makes CI keyless and reproducible. It is useful for API behavior tests, but its vectors are **not semantic embeddings**. For an offline query, use `minifugu::deterministic_embedding(query, dims)` to produce a vector in the same space.

For real semantic embeddings, opt in to OpenAI mode:

```sh
export MINIFUGU_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=... # load from your secret manager; never commit the value
cargo run
```

This sends text in native `embed` fields to OpenAI's `/v1/embeddings` endpoint using the model named in the schema. Symbolic currently uses `openai/text-embedding-3-small` with 1536 dimensions for its external chunks. The querying client must provide a vector produced by the same model. The endpoint can be changed with `MINIFUGU_OPENAI_BASE_URL` for a local test server. No provider call occurs when writing explicit vectors. See [OpenAI's embedding guide](https://developers.openai.com/api/docs/guides/embeddings).

## Supported API

| Endpoint or feature | Behavior |
| --- | --- |
| `POST /v2/namespaces/:namespace` | Atomic upserts, ID deletes, delete by filter, schema declarations, cosine distance metric, `rows_affected` |
| `POST /v2/namespaces/:namespace/query` | Single and multi-query responses, `rows`, `$dist`, `include_attributes`, `limit`, `offset` |
| `DELETE /v2/namespaces/:namespace` | Removes a namespace and all its rows |
| Filters | `And`, `Eq`, `NotEq`, `In`, `Gte`, `Lte`; RFC 3339 timestamps compare as instants |
| Ranking | Exact cosine for `ANN`; BM25 on full-text fields; `Sum` and weighted `Product` of BM25 clauses |
| Errors | JSON `{ "status": "error", "error": "..." }`; missing bearer token 401, missing namespace 404, invalid schema/query 400 |

The write endpoint accepts `uint`, `uuid`, `string`, `bool`, `datetime`, numeric, and fixed-length f16 vector schemas. It infers simple scalar fields when no declaration is supplied. Namespace state is isolated by name.

## Known differences

- Exact cosine replaces Turbopuffer's approximate ANN. This is appropriate for small fixtures, not performance comparisons.
- BM25 uses a simple Unicode alphanumeric tokenizer. It has no stemming, language-specific segmentation, or Turbopuffer index tuning, so scores and some ranks differ.
- Default native embeddings are deterministic hashes. OpenAI mode uses real embeddings but still has no Turbopuffer index behavior.
- State is in memory only. There is no disk persistence, sharding, cache, durability, rate limiting, or production-scale indexing.
- Unsupported operations include patching, aggregations, RRF server-side reranking, `Or`/`Not` filters, namespace metadata, and export. They are not needed by the current contract suite.
- Error wording is not byte-for-byte compatible. Tests should assert status and `status`/`error` shape, plus the relevant field name.

## Test

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`tests/compatibility.rs` runs the same disposable-namespace contract against MiniFugu and, when both environment variables are set, the real service:

```sh
TURBOPUFFER_BASE_URL=https://aws-us-west-2.turbopuffer.com \
TURBOPUFFER_API_KEY=<load-from-secret-manager> \
cargo test --test compatibility optional_real_turbopuffer_contract
```

The compatibility test deletes its namespace before asserting results. It writes only synthetic rows and uses a fresh random name. For a separate live OpenAI smoke test, set `MINIFUGU_LIVE_OPENAI=1` and `OPENAI_API_KEY`, then run `cargo test --test openai_live`. Ordinary CI runs have neither live-test flag and never call Turbopuffer or OpenAI.

## License

[MIT](LICENSE). MiniFugu is an independent test tool and is not affiliated with Turbopuffer.
