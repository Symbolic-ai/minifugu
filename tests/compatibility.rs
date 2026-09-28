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
        "schema":{"id":"uint","group":"string","tags":{"type":"[]string","full_text_search":true,"filterable":true},"score":"uint","title":{"type":"string","full_text_search":true,"glob":true,"regex":true},"vector":{"type":"[2]f32","ann":true}},
        "distance_metric":"cosine_distance",
        "upsert_columns":{"id":[1,2],"group":["a","a"],"tags":[["fugu"],["whale"]],"score":[3,5],"title":["small fugu","blue whale"],"vector":[[1,0],[0,1]]},
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
    let fused = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({
            "queries":[
                {"rank_by":["title","BM25","fugu"],"limit":2,"include_attributes":["title"]},
                {"rank_by":["title","BM25","whale"],"limit":2,"include_attributes":["title"]}
            ],
            "rerank_by":["RRF",{"weights":[2,1],"rank_constant":10}],
            "limit":{"total":2}
        }),
    )
    .await;
    let text_filter = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"filters":["title","ContainsAllTokens","small fu",{"last_as_prefix":true}],"limit":2}),
    ).await;
    let sequence_filter = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"filters":["title","ContainsTokenSequence","blue whale"],"limit":2}),
    ).await;
    let not_in = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"filters":["id","NotIn",[1]],"limit":2}),
    )
    .await;
    let computed = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"limit":2,"compute_attributes":{"fugu_score":["title","BM25","fugu"],"vector_distance":["vector","VectorDist",[1,0]]},"consistency":{"level":"strong"}}),
    )
    .await;
    let glob = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"filters":["title","IGlob","SMALL*"],"limit":2}),
    )
    .await;
    let regex = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["id","asc"],"filters":["title","Regex","^blue\\s+whale$"],"limit":2}),
    )
    .await;
    let knn = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["vector","kNN",[1,0]],"filters":["group","Eq","a"],"limit":2}),
    )
    .await;
    let attribute_rank = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["Attribute","score"],"limit":2}),
    )
    .await;
    let max_rank = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["Max",[["title","BM25","fugu"],["title","BM25","whale"]]],"limit":2}),
    )
    .await;
    let filter_rank = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["score","Gt",3],"limit":2}),
    )
    .await;
    let array_text = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["tags","BM25","fugu"],"filters":["tags","ContainsAnyToken","fugu"],"limit":2}),
    ).await;
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
    let product = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({"rank_by":["Product",["title","BM25","fugu"],2],"limit":2}),
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
    assert_eq!(fused.0, StatusCode::OK, "RRF response: {:?}", fused.1);
    assert_eq!(fused.1["results"][0]["rows"][0]["id"], 1);
    assert_eq!(fused.1["results"][0]["rows"][1]["id"], 2);
    assert!(fused.1["results"][0]["rows"][0]["$dist"].as_f64().unwrap() > 0.18);
    assert_eq!(
        text_filter.0,
        StatusCode::OK,
        "token filter response: {:?}",
        text_filter.1
    );
    assert_eq!(text_filter.1["rows"][0]["id"], 1);
    assert_eq!(
        sequence_filter.0,
        StatusCode::OK,
        "sequence filter response: {:?}",
        sequence_filter.1
    );
    assert_eq!(sequence_filter.1["rows"][0]["id"], 2);
    assert_eq!(not_in.0, StatusCode::OK);
    assert_eq!(not_in.1["rows"][0]["id"], 2);
    assert_eq!(
        computed.0,
        StatusCode::OK,
        "computed response: {:?}",
        computed.1
    );
    assert!(computed.1["rows"][0]["fugu_score"].as_f64().unwrap() > 0.0);
    assert_eq!(computed.1["rows"][0]["vector_distance"], 0.0);
    assert_eq!(glob.0, StatusCode::OK, "glob response: {:?}", glob.1);
    assert_eq!(glob.1["rows"][0]["id"], 1);
    assert_eq!(regex.0, StatusCode::OK, "regex response: {:?}", regex.1);
    assert_eq!(regex.1["rows"][0]["id"], 2);
    assert_eq!(knn.0, StatusCode::OK, "kNN response: {:?}", knn.1);
    assert_eq!(knn.1["rows"][0]["id"], 1);
    assert_eq!(
        attribute_rank.0,
        StatusCode::OK,
        "Attribute response: {:?}",
        attribute_rank.1
    );
    assert_eq!(attribute_rank.1["rows"][0]["id"], 2);
    assert_eq!(max_rank.0, StatusCode::OK, "Max response: {:?}", max_rank.1);
    assert_eq!(max_rank.1["rows"].as_array().unwrap().len(), 2);
    assert_eq!(
        filter_rank.0,
        StatusCode::OK,
        "filter rank response: {:?}",
        filter_rank.1
    );
    assert_eq!(filter_rank.1["rows"][0]["id"], 2);
    assert_eq!(
        array_text.0,
        StatusCode::OK,
        "array text response: {:?}",
        array_text.1
    );
    assert_eq!(array_text.1["rows"][0]["id"], 1);
    assert_eq!(
        product.0,
        StatusCode::OK,
        "Product response: {:?}",
        product.1
    );
    assert_eq!(product.1["rows"][0]["id"], 1);
    assert_eq!(copied.0, StatusCode::OK);
    assert_eq!(copied.1["rows_affected"], 2);
    assert!(copied.1.get("rows_upserted").is_none());
    assert_eq!(copied_query.1["rows"][0]["id"], 1);
    assert_eq!(copy_cleanup.status(), StatusCode::OK);
    assert_eq!(source_cleanup.status(), StatusCode::OK);
}

async fn grouping_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-grouping-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","grp":"string","gnum":"uint"},
            "upsert_rows":[
                {"id":1,"grp":"a","gnum":5},
                {"id":2,"grp":"a","gnum":10},
                {"id":3,"grp":"b","gnum":20},
                {"id":4,"grp":"b","gnum":20}
            ]
        }),
    )
    .await;
    let per_group = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["id","asc"],"limit":{"total":4,"per":{"attributes":["grp"],"limit":1}},
            "include_attributes":["grp"]
        }),
    )
    .await;
    let ordered = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":[["grp","asc"],["id","desc"]],"limit":4
        }),
    )
    .await;
    let grouped = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "aggregate_by":{"count":["Count"]},"group_by":["gnum"],"top_k":2
        }),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "grouping write: {:?}", write.1);
    assert_eq!(
        per_group.0,
        StatusCode::OK,
        "per-group response: {:?}",
        per_group.1
    );
    assert_eq!(
        per_group.1["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| &row["id"])
            .collect::<Vec<_>>(),
        vec![&json!(1), &json!(3)]
    );
    assert_eq!(
        ordered.0,
        StatusCode::OK,
        "ordering response: {:?}",
        ordered.1
    );
    assert_eq!(
        ordered.1["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| &row["id"])
            .collect::<Vec<_>>(),
        vec![&json!(2), &json!(1), &json!(4), &json!(3)]
    );
    assert_eq!(
        grouped.0,
        StatusCode::OK,
        "grouped response: {:?}",
        grouped.1
    );
    assert_eq!(
        grouped.1["aggregation_groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|group| &group["gnum"])
            .collect::<Vec<_>>(),
        vec![&json!(5), &json!(10)]
    );
    assert_eq!(cleanup.status(), StatusCode::OK);
}

async fn null_filter_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-null-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","n":"int","tags":"[]string"},
            "upsert_rows":[
                {"id":1,"n":null,"tags":null},
                {"id":2,"n":5,"tags":["x"]},
                {"id":3,"n":10,"tags":["y"]}
            ]
        }),
    )
    .await;
    let mut results = Vec::new();
    for filter in [
        json!(["n", "Lt", 5]),
        json!(["n", "Lte", 5]),
        json!(["tags", "NotContains", "x"]),
        json!(["tags", "NotContainsAny", ["x"]]),
    ] {
        let result = response(
            &client,
            token,
            &format!("{url}/query"),
            json!({
                "rank_by":["id","asc"],"filters":filter,"limit":10
            }),
        )
        .await;
        results.push(result);
    }
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "null fixture write: {:?}", write.1);
    for (result, expected) in
        results
            .iter()
            .zip([json!([1]), json!([1, 2]), json!([1, 3]), json!([1, 3])])
    {
        assert_eq!(
            result.0,
            StatusCode::OK,
            "null filter response: {:?}",
            result.1
        );
        let ids = result.1["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].clone())
            .collect::<Vec<_>>();
        assert_eq!(Value::Array(ids), expected);
    }
    assert_eq!(cleanup.status(), StatusCode::OK);
}

async fn conditional_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-conditional-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let initial = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","status":"string"},
            "upsert_rows":[{"id":1,"status":"new"},{"id":2,"status":"locked"}]
        }),
    )
    .await;
    let upsert = response(
        &client,
        token,
        &url,
        json!({
            "upsert_condition":["status","Eq","new"],
            "upsert_rows":[{"id":1,"status":"updated"},{"id":2,"status":"updated"}]
        }),
    )
    .await;
    let patch = response(
        &client,
        token,
        &url,
        json!({
            "patch_condition":["status","Eq","locked"],
            "patch_rows":[{"id":1,"status":"patched"},{"id":2,"status":"patched"}]
        }),
    )
    .await;
    let delete = response(
        &client,
        token,
        &url,
        json!({
            "delete_condition":["status","Eq","patched"],"deletes":[1,2]
        }),
    )
    .await;
    let after = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["id","asc"],"include_attributes":["status"],"limit":2
        }),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();

    assert_eq!(
        initial.0,
        StatusCode::OK,
        "initial response: {:?}",
        initial.1
    );
    assert_eq!(
        upsert.0,
        StatusCode::OK,
        "conditional upsert response: {:?}",
        upsert.1
    );
    assert_eq!(upsert.1["rows_affected"], 1);
    assert_eq!(
        patch.0,
        StatusCode::OK,
        "conditional patch response: {:?}",
        patch.1
    );
    assert_eq!(patch.1["rows_affected"], 1);
    assert_eq!(
        delete.0,
        StatusCode::OK,
        "conditional delete response: {:?}",
        delete.1
    );
    assert_eq!(delete.1["rows_affected"], 1);
    assert_eq!(after.0, StatusCode::OK);
    assert_eq!(after.1["rows"], json!([{"id":1,"status":"updated"}]));
    assert_eq!(cleanup.status(), StatusCode::OK);
}

#[tokio::test]
async fn local_contract() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, minifugu::router()).await.unwrap() });
    contract(&format!("http://{address}"), "dummy").await;
    extended_contract(&format!("http://{address}"), "dummy").await;
    grouping_contract(&format!("http://{address}"), "dummy").await;
    null_filter_contract(&format!("http://{address}"), "dummy").await;
    conditional_contract(&format!("http://{address}"), "dummy").await;
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
    grouping_contract(base.trim_end_matches('/'), &token).await;
    null_filter_contract(base.trim_end_matches('/'), &token).await;
    conditional_contract(base.trim_end_matches('/'), &token).await;
}
