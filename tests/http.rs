use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;

async fn server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, minifugu::router()).await.unwrap() });
    format!("http://{address}/v2/namespaces")
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
async fn rejects_unknown_document_selector_even_for_empty_namespace() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/external-documents");
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "schema": {"id":"uuid", "title":{"type":"string","full_text_search":true}},
        "upsert_rows": [{"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "title":"fugu"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, error) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by": ["title", "BM25", "fish"],
            "filters": ["document_id", "In", ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]],
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error["status"], "error");
    assert!(error["error"].as_str().unwrap().contains("document_id"));
}

#[tokio::test]
async fn upserts_queries_deletes_and_namespaces_are_isolated() {
    let base = server().await;
    let client = Client::new();
    let a = format!("{base}/a");
    let b = format!("{base}/b");
    let schema = json!({
        "id":"uint", "document_id":"string", "status":"string",
        "title":{"type":"string","full_text_search":true},
        "vector":{"type":"[2]f16","ann":true}
    });
    let rows = json!([
        {"id":1,"document_id":"one","status":"active","title":"yellow fugu","vector":[1.0,0.0]},
        {"id":2,"document_id":"two","status":"active","title":"blue whale","vector":[0.0,1.0]}
    ]);
    let (status, result) = post(
        &client,
        &a,
        json!({"schema":schema,"distance_metric":"cosine_distance","upsert_rows":rows}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows_affected"], 2);
    let query = json!({"queries":[
        {"rank_by":["vector","ANN",[1.0,0.0]],"filters":["status","Eq","active"],"limit":2},
        {"rank_by":["title","BM25","fugu"],"include_attributes":["document_id"],"limit":2}
    ]});
    let (status, result) = post(&client, &format!("{a}/query"), query).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["results"][0]["rows"][0]["id"], 1);
    assert_eq!(result["results"][1]["rows"][0]["document_id"], "one");
    let (status, _) = post(
        &client,
        &format!("{b}/query"),
        json!({"rank_by":["id","asc"],"limit":1}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, result) = post(
        &client,
        &a,
        json!({"delete_by_filter":["document_id","Eq","one"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows_affected"], 1);
    let (status, result) = post(
        &client,
        &format!("{a}/query"),
        json!({"rank_by":["id","asc"],"limit":10}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"].as_array().unwrap().len(), 1);
    assert_eq!(result["rows"][0]["id"], 2);
}

#[tokio::test]
async fn failed_write_does_not_partially_change_rows() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/atomic");
    post(
        &client,
        &ns,
        json!({"schema":{"id":"uint","tag":"string"},"upsert_rows":[{"id":1,"tag":"old"}]}),
    )
    .await;
    let (status, _) = post(
        &client,
        &ns,
        json!({"deletes":[1],"upsert_rows":[{"id":2,"tag":false}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, result) = post(
        &client,
        &format!("{ns}/query"),
        json!({"rank_by":["id","asc"],"limit":10,"include_attributes":true}),
    )
    .await;
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(result["rows"][0]["tag"], "old");
}

#[tokio::test]
async fn native_embedding_is_deterministic_and_filters_include_date_bounds() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/native");
    let id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    let (status, _) = post(&client, &ns, json!({"schema":{
        "id":"uuid", "published_at":"datetime", "content":{"type":"string","full_text_search":true,"embed":{"model":"test","dims":4}}
    },"distance_metric":"cosine_distance","upsert_rows":[{"id":id,"content":"red fugu","published_at":"2026-01-01T01:00:00+01:00"}]})).await;
    assert_eq!(status, StatusCode::OK);
    let query = json!({"rank_by":["embed_content","ANN",[1.0,0.0,0.0,0.0]],"filters":["And",[
        ["published_at","NotEq",null], ["published_at","Gte","2026-01-01T00:00:00Z"]
    ]],"limit":1});
    let (_, first) = post(&client, &format!("{ns}/query"), query.clone()).await;
    let (_, second) = post(&client, &format!("{ns}/query"), query).await;
    assert_eq!(first, second);
    assert_eq!(first["rows"][0]["id"], id);
}

#[tokio::test]
async fn bearer_is_required_and_errors_have_api_shape() {
    let base = server().await;
    let response = Client::new()
        .post(format!("{base}/auth"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["status"], "error");
    assert!(body["error"].is_string());
}
