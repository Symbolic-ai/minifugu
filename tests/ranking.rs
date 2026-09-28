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
