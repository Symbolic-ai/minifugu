//! Optional live test. Set MINIFUGU_LIVE_OPENAI=1 and OPENAI_API_KEY.
use minifugu::EmbeddingMode;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

#[tokio::test]
async fn native_openai_embedding_can_be_queried_with_the_same_model() {
    if std::env::var("MINIFUGU_LIVE_OPENAI").as_deref() != Ok("1") {
        return;
    }
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            minifugu::router_with_mode(EmbeddingMode::OpenAI {
                api_key: key.clone(),
                base_url: "https://api.openai.com".into(),
            }),
        )
        .await
        .unwrap();
    });
    let client = Client::new();
    let ns = format!("http://{address}/v2/namespaces/live-openai");
    let write = client.post(&ns).bearer_auth("dummy").json(&json!({
        "schema":{"id":"uint","content":{"type":"string","full_text_search":true,"embed":{"model":"openai/text-embedding-3-small","dims":1536}}},
        "distance_metric":"cosine_distance",
        "upsert_rows":[{"id":1,"content":"tiny orange pufferfish"}]
    })).send().await.unwrap();
    let write_status = write.status();
    let write_body: Value = write.json().await.unwrap();
    assert!(
        write_status.is_success(),
        "write status {write_status}: {write_body}"
    );
    let key = std::env::var("OPENAI_API_KEY").unwrap();
    let embedding: Value = client.post("https://api.openai.com/v1/embeddings")
        .bearer_auth(key)
        .json(&json!({"model":"text-embedding-3-small","input":"tiny orange pufferfish","encoding_format":"float"}))
        .send().await.unwrap().json().await.unwrap();
    let vector = embedding
        .pointer("/data/0/embedding")
        .expect("OpenAI embedding response");
    let result: Value = client
        .post(format!("{ns}/query"))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["embed_content","ANN",vector],"limit":1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(result["rows"][0]["id"], 1);
    assert!(result["rows"][0]["$dist"].as_f64().unwrap().abs() < 0.000001);
}
