use minifugu::{router_with_data_dir, EmbeddingMode};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::path::Path;
use tokio::{net::TcpListener, task::JoinHandle};

async fn serve(directory: &Path) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = router_with_data_dir(EmbeddingMode::Deterministic, directory).unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{address}/v2/namespaces/persisted"), task)
}

#[tokio::test]
async fn rows_survive_restarts_and_deletes_are_durable() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    let write = client
        .post(&url)
        .bearer_auth("dummy")
        .json(&json!({
            "schema":{"id":"uint","title":{"type":"string","full_text_search":true}},
            "upsert_rows":[{"id":1,"title":"persistent fugu"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);
    let metadata_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
    let metadata_before: Value = client
        .get(&metadata_url)
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    task.abort();

    let (url, task) = serve(directory.path()).await;
    let result: Value = client
        .post(format!("{url}/query"))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["title","BM25","fugu"],"limit":1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(result["rows"][0]["id"], 1);
    let metadata_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
    let metadata_after: Value = client
        .get(&metadata_url)
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(metadata_after["created_at"], metadata_before["created_at"]);
    assert_eq!(metadata_after["updated_at"], metadata_before["updated_at"]);
    let deleted = client
        .delete(&url)
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
    task.abort();

    let (url, task) = serve(directory.path()).await;
    let query = client
        .post(format!("{url}/query"))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["id","asc"],"limit":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(query.status(), StatusCode::NOT_FOUND);
    task.abort();
}

#[test]
fn corrupt_data_fails_startup_instead_of_silently_erasing_rows() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("namespaces.json"), b"not json").unwrap();
    assert!(router_with_data_dir(EmbeddingMode::Deterministic, directory.path()).is_err());
}

#[tokio::test]
async fn older_snapshot_without_cached_byte_count_still_queries() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    let write = client
        .post(&url)
        .bearer_auth("dummy")
        .json(&json!({
            "schema":{"id":"uint","title":"string"},
            "upsert_rows":[{"id":1,"title":"old snapshot"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);
    task.abort();

    let path = directory.path().join("namespaces.json");
    let mut snapshot: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    snapshot["persisted"]
        .as_object_mut()
        .unwrap()
        .remove("approx_logical_bytes");
    std::fs::write(&path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

    let (url, task) = serve(directory.path()).await;
    let query = client
        .post(format!("{url}/query"))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["id","asc"],"limit":1}))
        .send()
        .await
        .unwrap();
    assert_eq!(query.status(), StatusCode::OK);
    let body: Value = query.json().await.unwrap();
    assert_eq!(body["rows"][0]["id"], 1);
    assert!(
        body["billing"]["billable_logical_bytes_queried"]
            .as_u64()
            .unwrap()
            > 0
    );
    task.abort();
}
