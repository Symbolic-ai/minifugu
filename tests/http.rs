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

#[tokio::test]
async fn patches_filters_and_namespace_inspection_work() {
    let base = server().await;
    let client = Client::new();
    let origin = base.trim_end_matches("/v2/namespaces");
    let ns = format!("{base}/inspection");
    let (status, result) = post(
        &client,
        &ns,
        json!({
            "schema":{"id":"uint","tag":"string","count":"uint"},
            "upsert_rows":[{"id":1,"tag":"fugu","count":1},{"id":2,"tag":"fish","count":2}],
            "return_affected_ids":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["status"], "OK");
    assert_eq!(result["upserted_ids"], json!([1, 2]));

    let (status, result) = post(
        &client,
        &ns,
        json!({
            "patch_rows":[{"id":1,"count":3}],
            "patch_by_filter":{"filters":["tag","Eq","fish"],"patch":{"count":4}},
            "return_affected_ids":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows_patched"], 2);
    assert_eq!(result["patched_ids"], json!([2, 1]));

    let (status, result) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "filters":["Or",[["tag","Eq","fugu"],["Not",["count","Lte",3]]]],
            "top_k":2,"exclude_attributes":["tag"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"].as_array().unwrap().len(), 2);
    assert!(result["rows"][0].get("tag").is_none());

    let response = client
        .get(format!("{origin}/v1/namespaces"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["namespaces"][0]["id"], "inspection");

    let url = format!("{origin}/v2/namespaces/inspection/metadata");
    let response = client.get(url).bearer_auth("dummy").send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["approx_row_count"], 2);
    assert_eq!(body["schema"]["tag"], "string");

    let url = format!("{origin}/v1/namespaces/inspection/schema");
    let response = client.get(&url).bearer_auth("dummy").send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response = client
        .post(&url)
        .bearer_auth("dummy")
        .json(&json!({"new":"string"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["new"], "string");
}

#[tokio::test]
async fn unsupported_fields_fail_loudly() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/unsupported");
    let (status, body) = post(&client, &ns, json!({"upsert_columns":{"id":[1]}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["status"], "error");
    post(
        &client,
        &ns,
        json!({"upsert_rows":[{"id":1,"title":"fugu"}]}),
    )
    .await;
    let (status, body) = post(&client, &ns, json!({"upsert_columns":{"id":[2]}})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("upsert_columns"));
    let (status, body) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["id","asc"],"limit":1,"aggregate_by":{"count":["Count"]}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("aggregate_by"));
}
