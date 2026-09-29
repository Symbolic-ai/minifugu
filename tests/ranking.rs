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
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"title":{"type":"string","full_text_search":{"stemming":true}}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
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
