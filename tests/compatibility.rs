//! Run with TURBOPUFFER_BASE_URL and TURBOPUFFER_API_KEY to compare the same
//! disposable-namespace contract with the live service. No account is needed
//! for the ordinary test run.
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use uuid::Uuid;

async fn response(client: &Client, token: &str, url: &str, body: Value) -> (StatusCode, Value) {
    let response = client
        .post(url)
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.json().await.unwrap();
    (status, body)
}

async fn contract(base: &str, token: &str) {
    let client = Client::new();
    let namespace = format!("minifugu-compat-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{namespace}");
    let query_url = format!("{url}/query");
    let first = Uuid::new_v4().to_string();
    let second = Uuid::new_v4().to_string();
    let write = response(&client, token, &url, json!({
        "schema": {"id":"uuid", "title":{"type":"string","full_text_search":true},"status":"string"},
        "upsert_rows": [
            {"id":first,"title":"tiny orange fugu","status":"active"},
            {"id":second,"title":"blue whale","status":"inactive"}
        ]
    })).await;
    let bad_filter = response(
        &client,
        token,
        &query_url,
        json!({
            "rank_by":["title","BM25","fugu"],
            "filters":["document_id","Eq",first],
            "limit":10
        }),
    )
    .await;
    let multi = response(&client, token, &query_url, json!({"queries":[
        {"rank_by":["title","BM25","fugu"],"filters":["status","Eq","active"],"include_attributes":["title"],"limit":10},
        {"rank_by":["title","BM25","whale"],"limit":10}
    ]})).await;
    let deleted = response(&client, token, &url, json!({"deletes":[second]})).await;
    let after = response(
        &client,
        token,
        &query_url,
        json!({
            "rank_by":["title","BM25","whale"],"limit":10
        }),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();

    assert_eq!(write.0, StatusCode::OK, "write response: {:?}", write.1);
    assert_eq!(write.1["rows_affected"], 2);
    assert_eq!(
        bad_filter.0,
        StatusCode::BAD_REQUEST,
        "unknown-field response: {:?}",
        bad_filter.1
    );
    assert_eq!(bad_filter.1["status"], "error");
    assert_eq!(
        multi.0,
        StatusCode::OK,
        "multi-query response: {:?}",
        multi.1
    );
    assert_eq!(multi.1["results"][0]["rows"][0]["id"], first);
    assert_eq!(
        multi.1["results"][0]["rows"][0]["title"],
        "tiny orange fugu"
    );
    assert_eq!(multi.1["results"][1]["rows"][0]["id"], second);
    assert_eq!(deleted.0, StatusCode::OK);
    assert_eq!(after.1["rows"], json!([]));
    assert_eq!(cleanup.status(), StatusCode::OK);
}

#[tokio::test]
async fn local_contract() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, minifugu::router()).await.unwrap() });
    contract(&format!("http://{address}"), "dummy").await;
}

#[tokio::test]
async fn optional_real_turbopuffer_contract() {
    let (Ok(base), Ok(token)) = (
        std::env::var("TURBOPUFFER_BASE_URL"),
        std::env::var("TURBOPUFFER_API_KEY"),
    ) else {
        return;
    };
    contract(base.trim_end_matches('/'), &token).await;
}
