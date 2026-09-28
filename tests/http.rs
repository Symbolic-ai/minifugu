use base64::{engine::general_purpose::STANDARD, Engine};
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
    let fused = json!({
        "queries":[
            {"rank_by":["vector","ANN",[1.0,0.0]],"limit":2},
            {"rank_by":["title","BM25","fugu"],"limit":2,"include_attributes":["document_id"]}
        ],
        "rerank_by":["RRF",{"rank_constant":10,"weights":[2,1]}],
        "limit":{"total":1}
    });
    let (status, result) = post(&client, &format!("{a}/query"), fused.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["results"][0]["rows"][0]["id"], 1);
    assert_eq!(result["results"][0]["rows"][0]["document_id"], "one");
    assert!((result["results"][0]["rows"][0]["$dist"].as_f64().unwrap() - 3.0 / 11.0).abs() < 1e-9);
    let mut invalid = fused;
    invalid["rerank_by"][1]["weights"] = json!([1]);
    let (status, error) = post(&client, &format!("{a}/query"), invalid).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"].as_str().unwrap().contains("weights"));
    let (status, computed) = post(
        &client,
        &format!("{a}/query"),
        json!({"rank_by":["id","asc"],"limit":2,"compute_attributes":{"distance":["vector","VectorDist",[1,0]]}}),
    ).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(computed["rows"][0]["distance"], 0.0);
    assert_eq!(computed["rows"][1]["distance"], 1.0);
    let (status, tokens) = post(
        &client,
        &format!("{a}/query"),
        json!({"rank_by":["id","asc"],"filters":["title","ContainsAnyToken",["whal","fug"],{"last_as_prefix":true}],"limit":2}),
    ).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tokens["rows"].as_array().unwrap().len(), 1);
    assert_eq!(tokens["rows"][0]["id"], 1);
    let (status, error) = post(
        &client,
        &format!("{a}/query"),
        json!({"rank_by":["Sum",[["vector","kNN",[1,0]]]],"limit":2}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"]
        .as_str()
        .unwrap()
        .contains("kNN requires filters"));
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
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "patch_by_filter":{"filters":["id","Eq",1],"patch":{"id":3}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, result) = post(
        &client,
        &format!("{ns}/query"),
        json!({"rank_by":["id","asc"],"limit":10,"include_attributes":true}),
    )
    .await;
    assert_eq!(result["rows"], json!([{"id":1,"tag":"old"}]));
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

    let url = format!("{origin}/v1/namespaces/inspection/metadata");
    let response = client.get(url).bearer_auth("dummy").send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["approx_row_count"], 2);
    assert_eq!(body["schema"]["tag"], "string");
    assert!(body["approx_logical_bytes"].as_u64().unwrap() > 0);
    assert!(body["created_at"].as_str().unwrap().ends_with('Z'));
    assert!(body["updated_at"].as_str().unwrap().ends_with('Z'));
    assert_eq!(body["encryption"], json!({"sse":true}));
    assert_eq!(body["index"]["status"], "up-to-date");

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
async fn documented_cache_recall_and_explain_routes_work() {
    let base = server().await;
    let client = Client::new();
    let origin = base.trim_end_matches("/v2/namespaces");
    let ns = format!("{base}/operations");
    let (status, _) = post(&client, &ns, json!({
        "schema":{"id":"uint","vector":{"type":"[2]f32","ann":true},"tag":"string"},
        "distance_metric":"cosine_distance",
        "upsert_rows":[{"id":1,"vector":[1,0],"tag":"fish"},{"id":2,"vector":[0,1],"tag":"other"}]
    })).await;
    assert_eq!(status, StatusCode::OK);

    let warm = client
        .get(format!("{origin}/v1/namespaces/operations/hint_cache_warm"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap();
    assert_eq!(warm.status(), StatusCode::ACCEPTED);
    let warm_body: Value = warm.json().await.unwrap();
    assert_eq!(warm_body["status"], "ACCEPTED");

    let (status, recall) = post(
        &client,
        &format!("{origin}/v1/namespaces/operations/_debug/recall"),
        json!({"num":2,"top_k":1,"include_ground_truth":true,"filters":["tag","Eq","fish"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(recall["avg_recall"], 1.0);
    assert_eq!(recall["avg_exhaustive_count"], 1.0);
    assert_eq!(recall["ground_truth"].as_array().unwrap().len(), 2);
    assert_eq!(recall["ground_truth"][0]["nearest_neighbors"][0]["id"], 1);

    let (status, explanation) = post(
        &client,
        &format!("{ns}/explain_query"),
        json!({"rank_by":["vector","ANN",[1,0]],"limit":1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(explanation["plan_text"].as_str().unwrap().contains("exact"));

    let (status, invalid) = post(
        &client,
        &format!("{ns}/explain_query"),
        json!({"rank_by":["absent","ANN",[1,0]],"limit":1}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(invalid["error"].as_str().unwrap().contains("absent"));
}

#[tokio::test]
async fn base64_vectors_round_trip_through_write_and_query() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/encoded");
    let encoded = STANDARD.encode([1.0_f32.to_le_bytes(), 0.0_f32.to_le_bytes()].concat());
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "schema":{"id":"uint","vector":{"type":"[2]f32","ann":true}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"vector":encoded}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, response) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["vector","ANN",encoded],"limit":1,
            "include_attributes":["vector"],"vector_encoding":"base64"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["rows"][0]["id"], 1);
    assert_eq!(response["rows"][0]["vector"], encoded);
    assert!(
        response["billing"]["billable_logical_bytes_queried"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(response["performance"]["approx_namespace_size"], 1);

    let (status, response) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["vector","ANN",[1,0]],"limit":1,
            "include_attributes":["vector"],"vector_encoding":"float"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["rows"][0]["vector"], json!([1.0, 0.0]));
}

#[tokio::test]
async fn euclidean_metric_and_i8_vectors_rank_exactly() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/metric");
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "schema":{"id":"uint","vector":{"type":"[2]i8","ann":true}},
            "distance_metric":"euclidean_squared",
            "upsert_rows":[{"id":1,"vector":[0,0]},{"id":2,"vector":[3,4]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, response) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["vector","ANN",[1,1]],"limit":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["rows"][0]["id"], 1);
    assert_eq!(response["rows"][0]["$dist"], 2.0);
    assert_eq!(response["rows"][1]["$dist"], 13.0);
}

#[tokio::test]
async fn partial_write_flags_finish_small_local_filters() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/partial");
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "schema":{"id":"uint","tag":"string"},
            "upsert_rows":[{"id":1,"tag":"old"},{"id":2,"tag":"old"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, patched) = post(
        &client,
        &ns,
        json!({
            "patch_by_filter":{"filters":["tag","Eq","old"],"patch":{"tag":"new"}},
            "patch_by_filter_allow_partial":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(patched["rows_patched"], 2);
    assert!(patched.get("rows_remaining").is_none());
    let (status, deleted) = post(
        &client,
        &ns,
        json!({
            "delete_by_filter":["tag","Eq","new"],
            "delete_by_filter_allow_partial":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["rows_deleted"], 2);
    assert!(deleted.get("rows_remaining").is_none());
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "patch_by_filter":{"filters":["tag","Eq","new"],"patch":{"tag":"old"}},
            "disable_backpressure":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unsupported_fields_fail_loudly() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/unsupported");
    let (status, body) = post(&client, &ns, json!({"unknown_write_option":true})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["status"], "error");
    post(
        &client,
        &ns,
        json!({"upsert_rows":[{"id":1,"title":"fugu"}]}),
    )
    .await;
    let (status, body) = post(&client, &ns, json!({"upsert_condition":["id","Eq",2]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("upsert_condition"));
    let (status, body) = post(
        &client,
        &ns,
        json!({"schema":{"title":{"type":"string","fuzzy":true}}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("unsupported schema option fuzzy"));
    let (status, body) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["id","asc"],"limit":1,"aggregate_by":{"count":["Count"]}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("aggregation"));
    for (query, expected) in [
        (
            json!({"rank_by":["id","asc"],"top_k":{"total":1,"per":{"attributes":["title"],"limit":1}},"include_attributes":["title"]}),
            "top_k must",
        ),
        (
            json!({"rank_by":["id","asc"],"limit":{"total":1,"extra":true}}),
            "unsupported limit field",
        ),
        (
            json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":["title"],"limit":1,"extra":true}},"include_attributes":["title"]}),
            "unsupported limit.per field",
        ),
        (
            json!({"aggregate_by":{"count":["Count"]},"top_k":1}),
            "top_k requires group_by",
        ),
        (
            json!({"aggregate_by":{"title":["Count"]},"group_by":["title"],"top_k":1}),
            "conflicts with a group field",
        ),
    ] {
        let (status, body) = post(&client, &format!("{ns}/query"), query).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"].as_str().unwrap().contains(expected), "{body}");
    }
    let (status, body) = post(
        &client,
        &format!("{base}/bad-columns"),
        json!({"upsert_columns":{"title":["x"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("columns require an id array"));
}

#[tokio::test]
async fn column_writes_array_filters_and_group_limits_work() {
    let base = server().await;
    let client = Client::new();
    let ns = format!("{base}/columns");
    let (status, write) = post(
        &client,
        &ns,
        json!({
        "schema":{"id":"uint","group":"string","tags":"[]string","score":"uint","vector":{"type":"[2]f32","ann":true}},
            "distance_metric":"cosine_distance",
            "upsert_columns":{
                "id":[1,2,3],"group":["a","a","b"],
            "tags":[["fugu","fish"],["whale"],["fugu"]],"score":[3,5,7],
                "vector":[[1,0],[0,1],[1,1]]
            },
            "return_affected_ids":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(write["upserted_ids"], json!([1, 2, 3]));

    let (status, patch) = post(
        &client,
        &ns,
        json!({
            "patch_columns":{"id":[2],"tags":[["fugu","whale"]]},
            "return_affected_ids":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(patch["patched_ids"], json!([2]));

    let (status, result) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["id","asc"],"filters":["And",[
                ["tags","Contains","fugu"],["id","Gt",0],["id","Lt",4]
            ]],"limit":{"total":3,"per":{"attributes":["group"],"limit":1}},
            "include_attributes":["tags","group"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"].as_array().unwrap().len(), 2);
    assert_eq!(result["rows"][0]["id"], 1);
    assert_eq!(result["rows"][1]["id"], 3);
    let (status, any_gt) = post(
        &client,
        &format!("{ns}/query"),
        json!({"rank_by":["id","asc"],"filters":["tags","AnyGt","v"],"limit":3}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(any_gt["rows"].as_array().unwrap().len(), 1);
    assert_eq!(any_gt["rows"][0]["id"], 2);

    let (status, error) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["id","asc"],
            "limit":{"total":3,"per":{"attributes":["group"],"limit":1}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["error"].as_str().unwrap().contains("group"));

    let (status, result) = post(&client, &format!("{ns}/query"), json!({
        "rank_by":["vector","ANN",[0.0,1.0]],"filters":["tags","ContainsAny",["whale"]],"top_k":1
    })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"][0]["id"], 2);

    let (status, aggregate) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "aggregate_by":{"count":["Count"],"sum":["Sum","score"]}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(aggregate["aggregations"], json!({"count":3,"sum":15}));
    let (status, grouped) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "aggregate_by":{"count":["Count"],"sum":["Sum","score"]},
            "group_by":["group"],"top_k":2
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        grouped["aggregation_groups"],
        json!([
            {"group":"a","count":2,"sum":8},{"group":"b","count":1,"sum":7}
        ])
    );
    let (status, ordered) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":[["group","asc"],["id","desc"]],"limit":3
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        ordered["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![2, 1, 3]
    );
}

#[tokio::test]
async fn namespaces_can_be_copied_and_then_diverge() {
    let base = server().await;
    let client = Client::new();
    let source = format!("{base}/source");
    let branch = format!("{base}/branch");
    let copy = format!("{base}/copy");
    post(
        &client,
        &source,
        json!({"upsert_rows":[{"id":1,"title":"fugu"}]}),
    )
    .await;
    let (status, result) = post(&client, &branch, json!({"branch_from_namespace":"source"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows_affected"], 1);
    let (status, _) = post(
        &client,
        &copy,
        json!({"copy_from_namespace":{"source_namespace":"source"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    post(
        &client,
        &branch,
        json!({"upsert_rows":[{"id":2,"title":"branch only"}]}),
    )
    .await;
    let (_, source_rows) = post(
        &client,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"limit":10}),
    )
    .await;
    let (_, branch_rows) = post(
        &client,
        &format!("{branch}/query"),
        json!({"rank_by":["id","asc"],"limit":10}),
    )
    .await;
    let (_, copy_rows) = post(
        &client,
        &format!("{copy}/query"),
        json!({"rank_by":["id","asc"],"limit":10}),
    )
    .await;
    assert_eq!(source_rows["rows"].as_array().unwrap().len(), 1);
    assert_eq!(copy_rows["rows"].as_array().unwrap().len(), 1);
    assert_eq!(branch_rows["rows"].as_array().unwrap().len(), 2);
    let (status, _) = post(&client, &branch, json!({"branch_from_namespace":"source"})).await;
    assert_eq!(status, StatusCode::CONFLICT);
    let origin = base.trim_end_matches("/v2/namespaces");
    let first: Value = client
        .get(format!("{origin}/v1/namespaces?page_size=2"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(first["namespaces"].as_array().unwrap().len(), 2);
    assert!(first["next_cursor"].is_string());
    let second: Value = client
        .get(format!(
            "{origin}/v1/namespaces?page_size=2&cursor={}",
            first["next_cursor"].as_str().unwrap()
        ))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(second["namespaces"].as_array().unwrap().len(), 1);
    let prefix: Value = client
        .get(format!("{origin}/v1/namespaces?prefix=bra"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(prefix["namespaces"][0]["id"], "branch");
}
