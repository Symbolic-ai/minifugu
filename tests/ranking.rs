use axum::{routing::post as axum_post, Json, Router};
use minifugu::{deterministic_embedding, EmbeddingMode};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;

async fn serve(router: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{address}")
}

async fn post(client: &Client, url: &str, body: Value) -> (StatusCode, Value) {
    let response = client
        .post(url)
        .bearer_auth("dummy")
        .json(&body)
        .send()
        .await
        .unwrap();
    (response.status(), response.json().await.unwrap())
}

#[tokio::test]
async fn equal_bm25_scores_order_numeric_ids_numerically() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/bm25-ties");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","title":{"type":"string","full_text_search":true}},
            "upsert_rows":[
                {"id":11,"title":"fugu"},{"id":5,"title":"fugu"},{"id":2,"title":"fugu"},
                {"id":9007199254740993_u64,"title":"fugu"},
                {"id":9007199254740992_u64,"title":"fugu"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["title","BM25","fugu"],"limit":5
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| &row["id"])
            .collect::<Vec<_>>(),
        vec![
            &json!(2),
            &json!(5),
            &json!(11),
            &json!(9007199254740992_u64),
            &json!(9007199254740993_u64)
        ]
    );
}

#[tokio::test]
async fn attribute_and_rrf_ties_order_numeric_ids_numerically() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/ranking-ties");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","group":"string"},
            "upsert_rows":[{"id":11,"group":"same"},{"id":2,"group":"same"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, ordered) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["group","asc"],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ids(&ordered), [2, 11]);
    let (status, fused) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "queries":[
                {"rank_by":["id","asc"],"limit":2},
                {"rank_by":["id","desc"],"limit":2}
            ],
            "rerank_by":["RRF"],"limit":2,"vector_encoding":"base64"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{fused}");
    let ids = fused["results"][0]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids, [2, 11]);
}

#[tokio::test]
async fn exact_cosine_ranks_explicit_vectors_and_rejects_wrong_dimensions() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/vectors");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","vector":{"type":"[2]f16","ann":true}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[
                {"id":1,"vector":[1.0,0.0]},
                {"id":2,"vector":[0.0,1.0]},
                {"id":3,"vector":[-1.0,0.0]}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["vector","ANN",[1.0,0.0]],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        result["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert!(
        result["rows"][0]["$dist"].as_f64().unwrap() < result["rows"][1]["$dist"].as_f64().unwrap()
    );
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["vector","ANN",[1.0]],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(result["error"].as_str().unwrap().contains("dimensions"));
}

#[tokio::test]
async fn bm25_ranks_term_frequency_and_excludes_non_matches() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/bm25");
    post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","title":{"type":"string","full_text_search":true}},
            "upsert_rows":[
                {"id":1,"title":"orange orange fugu"},
                {"id":2,"title":"orange fugu"},
                {"id":3,"title":"blue whale"}
            ]
        }),
    )
    .await;
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["title","BM25","orange"],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let rows = result["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], 1);
    assert_eq!(rows[1]["id"], 2);
    assert!(rows[0]["$dist"].as_f64().unwrap() > rows[1]["$dist"].as_f64().unwrap());
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["title","BM25","oran",{"last_as_prefix":true}],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"].as_array().unwrap().len(), 2);
    assert_eq!(result["rows"][0]["$dist"], 1.0);
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["title","BM25","oran",{"last_as_prefix":"yes"}],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn sparse_knn_ranks_dot_products_and_excludes_non_matches() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/sparse");
    let (status, _) = post(&client, &url, json!({
        "schema":{"id":"uint","terms":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}}},
        "upsert_rows":[
            {"id":1,"terms":{"fugu":1.0,"blue":0.25}},
            {"id":2,"terms":{"fugu":0.5}},
            {"id":3,"terms":{"whale":1.0}}
        ]
    })).await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["terms","SparseKNN",{"fugu":1.0}],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"].as_array().unwrap().len(), 2);
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(result["rows"][0]["$dist"], 1.0);
    assert_eq!(result["rows"][1]["id"], 2);
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["terms","SparseKNN",{"fugu":"invalid"}],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn late_interaction_vectors_sum_best_document_token_distances() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/multivector");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","tokens":{"type":"[][2]f32","ann":{"late_interaction":true}}},
            "distance_metric":"euclidean_squared",
            "upsert_rows":[
                {"id":1,"tokens":[[1.0,0.0],[0.0,1.0]]},
                {"id":2,"tokens":[[0.9,0.1],[0.9,0.1]]}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["tokens","ANN",[[1.0,0.0],[0.0,1.0]]],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(result["rows"][0]["$dist"], 0.0);
    assert_eq!(result["rows"][1]["id"], 2);
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["tokens","ANN",[[1.0]]],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    for rank in [json!(["tokens", "asc"]), json!(["tokens", "desc"])] {
        let (status, _) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":rank,"limit":2}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "rank {rank}");
    }
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({"filters":["tokens","Eq",[[1.0,0.0]]],"limit":2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // ANN needs the late-interaction index; exact kNN scans without one, as it does for
    // single vectors.
    let plain = format!("{base}/v2/namespaces/multivector-plain");
    let (status, _) = post(
        &client,
        &plain,
        json!({
            "schema":{"id":"uint","tokens":{"type":"[][2]f32"}},
            "distance_metric":"euclidean_squared",
            "upsert_rows":[{"id":1,"tokens":[[1.0,0.0]]},{"id":2,"tokens":[[0.0,1.0]]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = post(
        &client,
        &format!("{plain}/query"),
        json!({"rank_by":["tokens","ANN",[[1.0,0.0]]],"limit":2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, result) = post(
        &client,
        &format!("{plain}/query"),
        json!({"rank_by":["tokens","kNN",[[1.0,0.0]]],"filters":["id","Gte",0],"limit":2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ids(&result), [1, 2]);
    assert_eq!(result["rows"][0]["$dist"], 0.0);
}

#[tokio::test]
async fn fuzzy_filter_matches_edit_distance_with_case_option() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/fuzzy");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","name":{"type":"string","fuzzy":true}},
            "upsert_rows":[
                {"id":1,"name":"Small Pufferfish"},
                {"id":2,"name":"blue whale"},
                {"id":3,"name":"pufferfish are cute"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let options =
        json!({"max_edit_distance":[{"min_query_chars":6,"distance":1}],"case_sensitive":false});
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "filters":["name","Fuzzy","pufferfsh",options],"rank_by":["id","asc"],"limit":10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // The match may start and end anywhere in the value, as on the live service.
    assert_eq!(ids(&result), [1, 3]);
    for thresholds in [
        json!([{"min_query_chars":12,"distance":3}]),
        json!([{"min_query_chars":5,"distance":1}]),
        json!([]),
    ] {
        let (status, _) = post(
            &client,
            &format!("{url}/query"),
            json!({
                "filters":["name","Fuzzy","pufferfsh",{"max_edit_distance":thresholds}],"limit":10
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "thresholds {thresholds}");
    }
    let (status, result) = post(&client, &format!("{url}/query"), json!({
        "filters":["name","Fuzzy","pufferfsh",{"max_edit_distance":[{"min_query_chars":6,"distance":1}],"case_sensitive":true}],"limit":10
    })).await;
    assert_eq!(status, StatusCode::OK);
    // "Small Pufferfish" needs a second edit for the capital P when case matters.
    assert_eq!(ids(&result), [3]);
}

#[tokio::test]
async fn numeric_ranking_saturates_and_decays_distances() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/numeric-rank");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","clicks":"uint","published_at":"datetime"},
            "upsert_rows":[
                {"id":1,"clicks":100,"published_at":"2026-01-01T00:00:00Z"},
                {"id":2,"clicks":10,"published_at":"2026-01-02T00:00:00Z"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["Saturate",["Attribute","clicks"],{"midpoint":100}],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(result["rows"][0]["$dist"], 0.5);
    let (status, result) = post(&client, &format!("{url}/query"), json!({
        "rank_by":["Decay",["Dist",["Attribute","published_at"],"2026-01-01T00:00:00Z"],{"midpoint":"1d"}],"limit":2
    })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(result["rows"][0]["$dist"], 1.0);
    assert_eq!(result["rows"][1]["$dist"], 0.5);
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["Saturate",["Attribute","clicks"],{"midpoint":0}],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn bytes_round_trip_and_reject_invalid_base64_or_filtering() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/bytes");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","blob":"bytes"},
            "upsert_rows":[{"id":1,"blob":"AP8="}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["id","asc"],"limit":10,"include_attributes":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["blob"], "AP8=");
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "filters":["blob","Eq","AP8="],"limit":10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post(
        &client,
        &url,
        json!({"upsert_rows":[{"id":2,"blob":"invalid"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn full_text_object_config_tunes_bm25_and_rejects_unimplemented_options() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/tuned-fts");
    let (status, _) = post(&client, &url, json!({
        "schema":{"id":"uint","title":{"type":"string","full_text_search":{"k1":2.0,"b":0.0,"k3":8.0}}},
        "upsert_rows":[{"id":1,"title":"fugu fugu"},{"id":2,"title":"fugu"}]
    })).await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["title","BM25","fugu fugu"],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 1);
    // idf ln(1.2) * tf 2*3/(2+2) with b=0 * query weight 2*9/(2+8). The default k1=1.2 and
    // b=0.75 would give 0.412577 instead.
    assert_close(
        result["rows"][0]["$dist"].as_f64().unwrap(),
        1.2_f64.ln() * 1.5 * 1.8,
    );
    // Invalid combinations are semantic errors (400); unknown names are shape errors (422),
    // matching the live service.
    for (config, expected) in [
        (
            json!({"stemming":true,"case_sensitive":true}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"remove_stopwords":true,"language":"arabic"}),
            StatusCode::BAD_REQUEST,
        ),
        (json!({"max_token_length":0}), StatusCode::BAD_REQUEST),
        (
            json!({"language":"klingon"}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"tokenizer":"word_v9"}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let (status, body) = post(
            &client,
            &format!("{base}/v2/namespaces/fts-config-errors"),
            json!({
                "schema":{"title":{"type":"string","full_text_search":config}},
                "upsert_rows":[{"id":1,"title":"fugu"}]
            }),
        )
        .await;
        assert_eq!(status, expected, "config {config}: {body}");
    }
}

#[tokio::test]
async fn deterministic_native_vectors_can_be_queried_with_the_same_embedder() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/native-rank");
    post(&client, &url, json!({
        "schema":{"id":"uint","content":{"type":"string","full_text_search":true,"embed":{"model":"openai/text-embedding-3-small","dims":64}}},
        "distance_metric":"cosine_distance",
        "upsert_rows":[{"id":1,"content":"red fugu"},{"id":2,"content":"blue whale"}]
    })).await;
    let query_vector = deterministic_embedding("red fugu", 64);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["embed_content","ANN",query_vector],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 1);
    assert!(result["rows"][0]["$dist"].as_f64().unwrap().abs() < 0.00001);
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["embed_content","BM25","red"],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "filters":["embed_content","Eq",[1.0,0.0]],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn openai_mode_embeds_native_text_and_queries_it() {
    async fn embed(Json(body): Json<Value>) -> Json<Value> {
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["dimensions"], 4);
        let vector = if body["input"] == "fugu" {
            vec![1.0, 0.0, 0.0, 0.0]
        } else {
            vec![0.0, 1.0, 0.0, 0.0]
        };
        Json(json!({"data":[{"embedding":vector}]}))
    }
    let provider = serve(Router::new().route("/v1/embeddings", axum_post(embed))).await;
    let base = serve(minifugu::router_with_mode(EmbeddingMode::OpenAI {
        api_key: "test-key".into(),
        base_url: provider,
    }))
    .await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/openai");
    let (status, _) = post(&client, &url, json!({
        "schema":{"id":"uint","content":{"type":"string","full_text_search":true,"embed":{"model":"openai/text-embedding-3-small","dims":4}}},
        "distance_metric":"cosine_distance",
        "upsert_rows":[{"id":1,"content":"fugu"},{"id":2,"content":"whale"}]
    })).await;
    assert_eq!(status, StatusCode::OK);
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "rank_by":["embed_content","ANN",[1.0,0.0,0.0,0.0]],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 1);
}

fn ids(result: &Value) -> Vec<u64> {
    result["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_u64().unwrap())
        .collect()
}

fn scores(result: &Value) -> Vec<f64> {
    result["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["$dist"].as_f64().unwrap())
        .collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

/// Expected scores and orders come from the same fixture on the live service.
#[tokio::test]
async fn bm25_weights_repeated_query_terms_like_the_live_service() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/bm25-repeat");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","text":{"type":"string","full_text_search":true}},
            "upsert_rows":[
                {"id":1,"text":"fish fish swim"},
                {"id":2,"text":"fish tank water deep blue"},
                {"id":3,"text":"swim swim fish"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    for (query, expected_ids, expected) in [
        ("fish", [1, 3, 2], [0.19350067, 0.14426166, 0.11623961]),
        ("fish fish", [1, 3, 2], [0.34830117, 0.25967097, 0.20923129]),
        (
            "fish fish swim",
            [3, 1, 2],
            [0.9407541, 0.85607296, 0.20923129],
        ),
    ] {
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":["text","BM25",query],"limit":10}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ids(&result), expected_ids, "query {query}");
        for (actual, expected) in scores(&result).into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }
}

/// Expected rows come from the same fixture on the live service.
#[tokio::test]
async fn numeric_ranking_keeps_zero_scores_and_requires_nonnegative_clauses() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/numeric-edges");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{
                "id":"uint","n":"int","u":"uint","t":"datetime",
                "text":{"type":"string","full_text_search":true}
            },
            "upsert_rows":[
                {"id":1,"n":0,"u":3,"t":"2026-01-01T00:00:00Z","text":"fish"},
                {"id":2,"n":4,"u":0,"t":"2026-01-02T06:00:00Z","text":"whale"},
                {"id":3,"n":10,"u":9,"t":"2026-01-05T00:00:00Z","text":"fish fish"},
                {"id":4,"n":-3,"u":1,"t":"2026-01-02T00:00:00Z","text":"crab"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let signed = json!(["Attribute", "n"]);
    let unsigned = json!(["Attribute", "u"]);
    let text = json!(["text", "BM25", "fish"]);
    for (rank, expected_ids, expected) in [
        (
            json!(["Max", [0, signed]]),
            vec![3, 2, 1, 4],
            vec![10.0, 4.0, 0.0, 0.0],
        ),
        (json!(["Max", [2, text]]), vec![1, 3], vec![2.0, 2.0]),
        (
            json!(["Saturate", signed, {"midpoint":5}]),
            vec![3, 2, 1, 4],
            vec![10.0 / 15.0, 4.0 / 9.0, 0.0, 0.0],
        ),
        (
            json!(["Dist", signed, 5]),
            vec![4, 1, 3, 2],
            vec![8.0, 5.0, 5.0, 1.0],
        ),
        (
            json!(["Dist", unsigned, 3]),
            vec![3, 2, 4, 1],
            vec![6.0, 3.0, 2.0, 0.0],
        ),
        (
            json!(["Product", 2, unsigned]),
            vec![3, 1, 4, 2],
            vec![18.0, 6.0, 2.0, 0.0],
        ),
        (
            json!(["Decay", ["Dist", ["Attribute", "t"], "2026-01-02T00:00:00Z"], {"midpoint":"12h"}]),
            vec![4, 2, 1, 3],
            vec![1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 7.0],
        ),
        (
            json!(["Decay", ["Dist", ["Attribute", "t"], "2026-01-02T00:00:00Z"], {"midpoint":"500ms"}]),
            vec![4, 2, 1, 3],
            vec![
                1.0,
                500.0 / 21_600_500.0,
                500.0 / 86_400_500.0,
                500.0 / 259_200_500.0,
            ],
        ),
    ] {
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":rank,"limit":10}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "rank {rank}: {result}");
        assert_eq!(ids(&result), expected_ids, "rank {rank}");
        for (actual, expected) in scores(&result).into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["Sum",[unsigned, text]],"limit":10}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ids(&result), [3, 1, 4, 2]);
    for rank in [
        signed.clone(),
        json!(["Product", 2, signed]),
        json!(["Max", [signed]]),
        json!(["Max", [signed, -1]]),
        json!(["Max", 0, signed]),
        json!(["Sum", [0.5, text]]),
        json!(["Dist", signed, 5.5]),
        json!(["Decay", ["Dist", unsigned, 3], {"midpoint":"2s"}]),
    ] {
        let (status, _) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":rank,"limit":10}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "rank {rank}");
    }
}

const TOKENIZATION_TEXT: &str = "Puffy (🐡) visited a café, e.g. searching for clams, but left disappointed. The foxes' e-mail: fish@sea.com 3.14 don't U.S.A. 東京タワー supercalifragilisticexpialidociousandmorelett1234 naïve Ærø straße";
const TOKENIZATION_QUERIES: &[&str] = &[
    "puffy",
    "Puffy",
    "🐡",
    "visited",
    "visit",
    "a",
    "café",
    "cafe",
    "e.g",
    "e",
    "g",
    "searching",
    "search",
    "clams",
    "clam",
    "for",
    "but",
    "the",
    "The",
    "foxes",
    "fox",
    "e-mail",
    "mail",
    "email",
    "fish",
    "sea.com",
    "sea",
    "com",
    "fish@sea.com",
    "3.14",
    "3",
    "14",
    "don't",
    "don",
    "t",
    "u.s.a",
    "u",
    "usa",
    "東",
    "京",
    "東京",
    "タワー",
    "東京タワー",
    "supercalifragilisticexpialidociousandmorelett1234",
    "naïve",
    "naive",
    "ærø",
    "aero",
    "aro",
    "straße",
    "strasse",
    "strase",
    "disappoint",
    "left",
];

/// Replays a probe of the live service: each field holds the same text under a different
/// analyzer, and a BM25 query matches row 1 exactly when one of its tokens is in the text.
#[tokio::test]
async fn text_analysis_matches_live_tokenization() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/tokenization");
    let fields = [
        ("d", json!(true)),
        ("stem", json!({"stemming": true})),
        ("stop", json!({"remove_stopwords": true})),
        ("fold", json!({"ascii_folding": true})),
        ("cs", json!({"case_sensitive": true})),
        ("max4", json!({"max_token_length": 4})),
        ("max254", json!({"max_token_length": 254})),
        ("v0", json!({"tokenizer": "word_v0"})),
        ("v1", json!({"tokenizer": "word_v1"})),
        ("v2", json!({"tokenizer": "word_v2"})),
        ("v3", json!({"tokenizer": "word_v3"})),
        (
            "fr",
            json!({"language": "french", "stemming": true, "remove_stopwords": true}),
        ),
        ("stemfold", json!({"stemming": true, "ascii_folding": true})),
    ];
    let expected: &[(&str, &[&str])] = &[
        (
            "d",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "stem",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "visit",
                "a",
                "café",
                "e.g",
                "e",
                "searching",
                "search",
                "clams",
                "clam",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "fox",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "disappoint",
                "left",
            ][..],
        ),
        (
            "stop",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "café",
                "e.g",
                "e",
                "searching",
                "clams",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "fold",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "cafe",
                "e.g",
                "e",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "naive",
                "ærø",
                "aero",
                "straße",
                "strasse",
                "left",
            ][..],
        ),
        (
            "cs",
            &[
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "searching",
                "clams",
                "for",
                "but",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "straße",
                "left",
            ][..],
        ),
        (
            "max4",
            &[
                "🐡",
                "a",
                "café",
                "e.g",
                "e",
                "for",
                "but",
                "the",
                "The",
                "e-mail",
                "mail",
                "fish",
                "fish@sea.com",
                "3.14",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "ærø",
                "left",
            ][..],
        ),
        (
            "max254",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "supercalifragilisticexpialidociousandmorelett1234",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "v0",
            &[
                "puffy",
                "Puffy",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "g",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "sea",
                "com",
                "fish@sea.com",
                "3.14",
                "3",
                "14",
                "don't",
                "don",
                "t",
                "u.s.a",
                "u",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "v1",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "g",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "sea",
                "com",
                "fish@sea.com",
                "3.14",
                "3",
                "14",
                "don't",
                "don",
                "t",
                "u.s.a",
                "u",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "v2",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "g",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "sea",
                "com",
                "fish@sea.com",
                "3.14",
                "3",
                "14",
                "don't",
                "don",
                "t",
                "u.s.a",
                "u",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "v3",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "e.g",
                "e",
                "searching",
                "clams",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "fr",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "a",
                "café",
                "cafe",
                "e.g",
                "e",
                "searching",
                "clams",
                "clam",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "fox",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "ærø",
                "straße",
                "left",
            ][..],
        ),
        (
            "stemfold",
            &[
                "puffy",
                "Puffy",
                "🐡",
                "visited",
                "visit",
                "a",
                "café",
                "cafe",
                "e.g",
                "e",
                "searching",
                "search",
                "clams",
                "clam",
                "for",
                "but",
                "the",
                "The",
                "foxes",
                "fox",
                "e-mail",
                "mail",
                "fish",
                "sea.com",
                "fish@sea.com",
                "3.14",
                "don't",
                "u.s.a",
                "東",
                "京",
                "東京",
                "タワー",
                "東京タワー",
                "naïve",
                "naive",
                "ærø",
                "aero",
                "straße",
                "disappoint",
                "left",
            ][..],
        ),
    ];
    let mut schema = serde_json::Map::new();
    let mut matching = serde_json::Map::new();
    let mut other = serde_json::Map::new();
    schema.insert("id".into(), json!("uint"));
    matching.insert("id".into(), json!(1));
    other.insert("id".into(), json!(2));
    for (field, config) in &fields {
        schema.insert(
            (*field).into(),
            json!({"type":"string","full_text_search":config}),
        );
        matching.insert((*field).into(), json!(TOKENIZATION_TEXT));
        other.insert((*field).into(), json!("unrelated filler words here"));
    }
    let (status, body) = post(
        &client,
        &url,
        json!({"schema":schema,"upsert_rows":[matching, other]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    for (field, hits) in expected {
        let mut actual = Vec::new();
        for query in TOKENIZATION_QUERIES {
            let (status, result) = post(
                &client,
                &format!("{url}/query"),
                json!({"rank_by":[field,"BM25",query],"limit":2}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{field} {query}: {result}");
            if ids(&result).contains(&1) {
                actual.push(*query);
            }
        }
        assert_eq!(actual, *hits, "field {field}");
    }
    for (query, expected_ids) in [
        ("visited café", vec![]),
        ("visited a café", vec![1]),
        ("visited the café", vec![1]),
    ] {
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"filters":["stop","ContainsTokenSequence",query],"rank_by":["id","asc"],"limit":2}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ids(&result), expected_ids, "sequence {query}");
    }
}

/// Expected rows and scores come from the same fixture on the live service.
#[tokio::test]
async fn text_filters_score_as_rank_clauses() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/rank-filters");
    let (status, body) = post(
        &client,
        &url,
        json!({
            "schema":{
                "id":"uint",
                "t":{"type":"string","full_text_search":{"stemming":true,"language":"english"}},
                "g":{"type":"string","fuzzy":true,"glob":true,"regex":true}
            },
            "upsert_rows":[
                {"id":1,"t":"The Quick fox","g":"turbopuffer"},
                {"id":2,"t":"slow fox","g":"turbopufer inc"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let fuzzy = json!(["g","Fuzzy","turbopufer",{"max_edit_distance":[{"min_query_chars":6,"distance":1}],"case_sensitive":false}]);
    for (rank, expected_ids, expected) in [
        (
            json!(["Sum", [["g", "Glob", "*turbopufer*"], fuzzy]]),
            vec![2, 1],
            vec![2.0, 1.0],
        ),
        (json!(["g", "Glob", "*turbopufer*"]), vec![2], vec![1.0]),
        (json!(["g", "Regex", "turbo.*"]), vec![1, 2], vec![1.0, 1.0]),
        (
            json!(["t", "ContainsAllTokens", "quick"]),
            vec![1],
            vec![1.0],
        ),
        (
            json!(["t", "ContainsAnyToken", "quick slow"]),
            vec![1, 2],
            vec![1.0, 1.0],
        ),
    ] {
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":rank,"limit":5}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "rank {rank}: {result}");
        assert_eq!(ids(&result), expected_ids, "rank {rank}");
        assert_eq!(scores(&result), expected, "rank {rank}");
    }
}

#[tokio::test]
async fn highlights_match_the_live_service() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/live_highlight.json")).unwrap();
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/highlights");
    let (status, body) = post(&client, &url, fixture["write"].clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":case["rank_by"],"compute_attributes":{"h":case["highlight"]},"limit":3}),
        )
        .await;
        assert_eq!(status.as_u16(), case["status"], "{label}: {result}");
        let Some(expected) = case["rows"].as_array() else {
            continue;
        };
        let rows = result["rows"].as_array().unwrap();
        assert_eq!(rows.len(), expected.len(), "{label}: {result}");
        for (row, expected) in rows.iter().zip(expected) {
            assert_eq!(row["id"], expected["id"], "{label}");
            assert_eq!(row["h"], expected["h"], "{label} row {}", row["id"]);
            if let Some(score) = expected["$dist"].as_f64() {
                assert_close(row["$dist"].as_f64().unwrap(), score);
            }
        }
    }
}

#[tokio::test]
async fn explicit_fragment_rank_does_not_change_sibling_bm25_score() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/fragment-stats");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","body":{"type":"string","full_text_search":true}},
            "upsert_rows":[
                {"id":1,"body":"red fish. Fish swim."},
                {"id":2,"body":"blue whale. Fish swim."}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let query_url = format!("{url}/query");
    let score = json!(["body", "BM25", "fish"]);
    let (status, baseline) = post(
        &client,
        &query_url,
        json!({
            "rank_by":["id","asc"],"limit":2,
            "compute_attributes":{"score":score}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, highlighted) = post(
        &client,
        &query_url,
        json!({
            "rank_by":["id","asc"],"limit":2,
            "compute_attributes":{
                "highlight":["Highlight","body",{"rank_fragments_by":score}],
                "score":score
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{highlighted}");
    for (baseline, highlighted) in baseline["rows"]
        .as_array()
        .unwrap()
        .iter()
        .zip(highlighted["rows"].as_array().unwrap())
    {
        assert_eq!(highlighted["score"], baseline["score"]);
    }
}

#[tokio::test]
async fn pre_tokenized_array_queries_and_validation_match_live() {
    let base = serve(minifugu::router()).await;
    let client = Client::new();
    let url = format!("{base}/v2/namespaces/pre-tokenized");
    let (status, body) = post(&client, &url, json!({
        "schema":{"id":"uint","text":{"type":"[]string","full_text_search":{"tokenizer":"pre_tokenized_array"}}},
        "upsert_rows":[{"id":1,"text":["FoO","bar"]}]
    })).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let query_url = format!("{url}/query");
    let (status, result) = post(
        &client,
        &query_url,
        json!({
            "rank_by":["text","BM25",["FoO"]],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ids(&result), [1]);
    let (status, _) = post(
        &client,
        &query_url,
        json!({
            "rank_by":["text","BM25","FoO"],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post(&client, &query_url, json!({
        "rank_by":["id","asc"],"limit":2,
        "compute_attributes":{"h":["Highlight","text",{"rank_fragments_by":["text","BM25",["FoO"]]}]}
    })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    for config in [
        json!({"tokenizer":"pre_tokenized_array","stemming":true}),
        json!({"tokenizer":"pre_tokenized_array","remove_stopwords":true}),
        json!({"tokenizer":"pre_tokenized_array","case_sensitive":false}),
        json!({"tokenizer":"pre_tokenized_array","language":"english"}),
    ] {
        let (status, _) = post(
            &client,
            &format!("{url}-invalid"),
            json!({
                "schema":{"id":"uint","text":{"type":"[]string","full_text_search":config}},
                "upsert_rows":[{"id":1,"text":["FoO"]}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{config}");
    }
}
