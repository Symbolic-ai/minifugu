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
async fn read_only_metadata_survives_restart() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("dummy")
            .json(&json!({"upsert_rows":[{"id":1}]}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    let metadata_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
    assert_eq!(
        client
            .patch(&metadata_url)
            .bearer_auth("dummy")
            .json(&json!({"read_only":true}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    task.abort();
    let (url, task) = serve(directory.path()).await;
    let metadata_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
    let metadata: Value = client
        .get(&metadata_url)
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(metadata["read_only"], true);
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("dummy")
            .json(&json!({"upsert_rows":[{"id":2}]}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    task.abort();
}

#[tokio::test]
async fn older_snapshot_without_cached_byte_count_still_queries() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    // Snapshots from before the cached byte estimate have no `approx_logical_bytes`.
    let snapshot = json!({"persisted":{
        "schema":{"id":"uint","title":"string"},
        "rows":{"n:1":{"id":1,"title":"old snapshot"}},
        "distance_metric":null
    }});
    std::fs::write(
        directory.path().join("namespaces.json"),
        serde_json::to_vec(&snapshot).unwrap(),
    )
    .unwrap();

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

async fn post(client: &Client, url: &str, body: Value) -> StatusCode {
    client
        .post(url)
        .bearer_auth("dummy")
        .json(&body)
        .send()
        .await
        .unwrap()
        .status()
}

async fn ids(client: &Client, url: &str) -> Vec<u64> {
    let response: Value = client
        .post(format!("{url}/query"))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["id","asc"],"limit":100}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    response["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn writes_append_row_changes_and_replay_after_restart() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    let schema = json!({"id":"uint","title":{"type":"string","full_text_search":true}});
    let rows = json!([{"id":1,"title":"one"},{"id":2,"title":"two"},{"id":3,"title":"three"}]);
    assert_eq!(
        post(&client, &url, json!({"schema":schema,"upsert_rows":rows})).await,
        StatusCode::OK
    );
    assert_eq!(
        post(
            &client,
            &url,
            json!({"patch_rows":[{"id":2,"title":"patched fugu"}]})
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(
        post(&client, &url, json!({"deletes":[3]})).await,
        StatusCode::OK
    );
    task.abort();

    let log = std::fs::read_to_string(directory.path().join("namespaces.log")).unwrap();
    assert_eq!(log.lines().count(), 3);
    // Later records carry only the rows they changed.
    assert!(!log.lines().nth(1).unwrap().contains("\"one\""));
    assert!(log.lines().nth(2).unwrap().contains("\"delete\":[\"n:3\"]"));

    let (url, task) = serve(directory.path()).await;
    assert_eq!(ids(&client, &url).await, vec![1, 2]);
    // The replayed byte estimate equals that of the same rows written at once.
    let fresh = url.replace("/persisted", "/fresh");
    let final_rows = json!([{"id":1,"title":"one"},{"id":2,"title":"patched fugu"}]);
    assert_eq!(
        post(
            &client,
            &fresh,
            json!({"schema":schema,"upsert_rows":final_rows})
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(
        logical_bytes(&client, &url).await,
        logical_bytes(&client, &fresh).await
    );
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
    assert_eq!(result["rows"][0]["id"], 2);
    // A log smaller than the compaction minimum is kept rather than rewritten.
    assert!(!directory.path().join("namespaces.json").exists());
    task.abort();
}

#[tokio::test]
async fn an_unacknowledged_final_log_line_is_discarded() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    assert_eq!(
        post(&client, &url, json!({"upsert_rows":[{"id":1}]})).await,
        StatusCode::OK
    );
    task.abort();
    let mut log = std::fs::OpenOptions::new()
        .append(true)
        .open(directory.path().join("namespaces.log"))
        .unwrap();
    std::io::Write::write_all(&mut log, br#"{"op":"put","name":"persisted","meta":{"#).unwrap();

    let (url, task) = serve(directory.path()).await;
    assert_eq!(ids(&client, &url).await, vec![1]);
    // Startup cut the partial line, so the next record starts a clean line.
    assert_eq!(
        post(&client, &url, json!({"upsert_rows":[{"id":2}]})).await,
        StatusCode::OK
    );
    task.abort();

    let (url, task) = serve(directory.path()).await;
    assert_eq!(ids(&client, &url).await, vec![1, 2]);
    task.abort();
}

#[test]
fn a_corrupt_complete_log_line_fails_startup() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("namespaces.log"), b"not json\n").unwrap();
    assert!(router_with_data_dir(EmbeddingMode::Deterministic, directory.path()).is_err());
}

#[tokio::test]
async fn a_copied_namespace_survives_restart() {
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    let copy = url.replace("/persisted", "/copy");
    assert_eq!(
        post(&client, &url, json!({"upsert_rows":[{"id":1},{"id":2}]})).await,
        StatusCode::OK
    );
    assert_eq!(
        post(&client, &copy, json!({"copy_from_namespace":"persisted"})).await,
        StatusCode::OK
    );
    task.abort();

    let (url, task) = serve(directory.path()).await;
    assert_eq!(
        ids(&client, &url.replace("/persisted", "/copy")).await,
        vec![1, 2]
    );
    task.abort();
}

async fn logical_bytes(client: &Client, url: &str) -> u64 {
    let metadata: Value = client
        .get(url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata")
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    metadata["approx_logical_bytes"].as_u64().unwrap()
}

#[tokio::test]
async fn a_snapshot_written_by_v0_2_0_loads_and_accepts_writes() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::copy(
        "tests/fixtures/v0.2.0-namespaces.json",
        directory.path().join("namespaces.json"),
    )
    .unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    let legacy = url.replace("/persisted", "/legacy");
    let result: Value = client
        .post(format!("{legacy}/query"))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["vector","ANN",[1,0]],"limit":1}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(
        post(&client, &legacy, json!({"deletes":[2]})).await,
        StatusCode::OK
    );
    task.abort();

    let (url, task) = serve(directory.path()).await;
    assert_eq!(
        ids(&client, &url.replace("/persisted", "/legacy")).await,
        vec![1]
    );
    let strings: Value = client
        .post(format!(
            "{}/query",
            url.replace("/persisted", "/legacy-strings")
        ))
        .bearer_auth("dummy")
        .json(&json!({"rank_by":["id","asc"],"limit":10,"include_attributes":true}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(strings["rows"][0]["note"], "string id");
    task.abort();
}

#[cfg(unix)]
#[tokio::test]
async fn data_files_are_private() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let client = Client::new();
    let (url, task) = serve(directory.path()).await;
    assert_eq!(
        post(&client, &url, json!({"upsert_rows":[{"id":1}]})).await,
        StatusCode::OK
    );
    task.abort();
    let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(directory.path()), 0o700);
    assert_eq!(mode(&directory.path().join("namespaces.log")), 0o600);
}
