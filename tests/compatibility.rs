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

async fn extended_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-extended-{}", Uuid::new_v4().simple());
    let source = format!("{base}/v2/namespaces/{name}");
    let copy = format!("{source}-copy");
    let write = response(&client, token, &source, json!({
        "schema":{"id":"uint","group":"string","tags":"[]string","score":"uint","title":{"type":"string","full_text_search":true}},
        "upsert_columns":{"id":[1,2],"group":["a","a"],"tags":[["fugu"],["whale"]],"score":[3,5],"title":["small fugu","blue whale"]},
        "return_affected_ids":true
    })).await;
    let query = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({
            "rank_by":["id","asc"],"filters":["tags","Contains","fugu"],
            "limit":{"total":2,"per":{"attributes":["group"],"limit":1}},
            "include_attributes":["group"]
        }),
    )
    .await;
    let aggregate = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({
            "aggregate_by":{"total":["Count"],"sum":["Sum","score"]}
        }),
    )
    .await;
    let grouped = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({
            "aggregate_by":{"total":["Count"],"sum":["Sum","score"]},
            "group_by":["group"],"top_k":10
        }),
    )
    .await;
    let ordered = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({
            "rank_by":[["group","asc"],["id","desc"]],"limit":2
        }),
    )
    .await;
    let copied = response(&client, token, &copy, json!({"copy_from_namespace":name})).await;
    let copied_query = response(
        &client,
        token,
        &format!("{copy}/query"),
        json!({
            "rank_by":["title","BM25","fugu"],"limit":1
        }),
    )
    .await;
    let copy_cleanup = client
        .delete(&copy)
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    let source_cleanup = client
        .delete(&source)
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    assert_eq!(write.0, StatusCode::OK);
    assert_eq!(write.1["rows_affected"], 2);
    assert_eq!(write.1["upserted_ids"], json!([1, 2]));
    assert_eq!(query.0, StatusCode::OK);
    assert_eq!(query.1["rows"][0]["id"], 1);
    assert_eq!(aggregate.0, StatusCode::OK);
    assert_eq!(aggregate.1["aggregations"], json!({"total":2,"sum":8}));
    assert_eq!(grouped.0, StatusCode::OK);
    assert_eq!(grouped.1["aggregation_groups"][0]["group"], "a");
    assert_eq!(grouped.1["aggregation_groups"][0]["sum"], 8);
    assert_eq!(ordered.0, StatusCode::OK);
    assert_eq!(ordered.1["rows"][0]["id"], 2);
    assert_eq!(ordered.1["rows"][1]["id"], 1);
    assert_eq!(copied.0, StatusCode::OK);
    assert_eq!(copied_query.1["rows"][0]["id"], 1);
    assert_eq!(copy_cleanup.status(), StatusCode::OK);
    assert_eq!(source_cleanup.status(), StatusCode::OK);
}

#[tokio::test]
async fn local_contract() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, minifugu::router()).await.unwrap() });
    contract(&format!("http://{address}"), "dummy").await;
    extended_contract(&format!("http://{address}"), "dummy").await;
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
    extended_contract(base.trim_end_matches('/'), &token).await;
}
