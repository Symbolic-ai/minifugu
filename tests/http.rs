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
async fn metadata_read_only_blocks_writes_and_is_inherited_by_branches() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/read-only-source");
    let metadata_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
    let schema_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/schema";
    assert_eq!(
        post(
            &client,
            &url,
            json!({"upsert_rows":[{"id":1,"title":"fish"}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let before: Value = client
        .get(&metadata_url)
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(before.get("read_only").is_none());
    assert_eq!(
        client
            .post(&schema_url)
            .bearer_auth("dummy")
            .json(&json!({"title":{"type":"string","regex":true}}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    let after_schema: Value = client
        .get(&metadata_url)
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(after_schema["last_write_at"], before["last_write_at"]);
    let patched = client
        .patch(&metadata_url)
        .bearer_auth("dummy")
        .json(&json!({"read_only":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(patched.status(), StatusCode::OK);
    let metadata: Value = patched.json().await.unwrap();
    assert_eq!(metadata["read_only"], true);
    assert_eq!(metadata["updated_at"], after_schema["updated_at"]);
    assert_eq!(metadata["last_write_at"], before["last_write_at"]);
    let (status, error) = post(
        &client,
        &url,
        json!({"upsert_rows":[{"id":2,"title":"whale"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    assert_eq!(
        error["error"],
        "💔 Writes not permitted. This namespace is read-only."
    );
    assert_eq!(
        post(
            &client,
            &url,
            json!({"patch_rows":[{"id":1,"title":"whale"}]})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        client
            .post(&schema_url)
            .bearer_auth("dummy")
            .json(&json!({"title":{"type":"string","regex":true}}))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(
            &client,
            &format!("{base}/read-only-branch"),
            json!({"branch_from_namespace":"read-only-source"})
        )
        .await
        .0,
        StatusCode::OK
    );
    let branch_metadata: Value = client
        .get(metadata_url.replace("read-only-source", "read-only-branch"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(branch_metadata["read_only"], true);
    assert_eq!(branch_metadata["last_write_at"], before["last_write_at"]);
    let queried = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["id","asc"],"limit":10}),
    )
    .await;
    assert_eq!(queried.0, StatusCode::OK);
    assert_eq!(queried.1["rows"].as_array().unwrap().len(), 1);
    let cleared = client
        .patch(&metadata_url)
        .bearer_auth("dummy")
        .json(&json!({"read_only":false}))
        .send()
        .await
        .unwrap();
    assert_eq!(cleared.status(), StatusCode::OK);
    assert!(cleared
        .json::<Value>()
        .await
        .unwrap()
        .get("read_only")
        .is_none());
    assert_eq!(
        post(
            &client,
            &url,
            json!({"upsert_rows":[{"id":2,"title":"whale"}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let invalid = client
        .patch(&metadata_url)
        .bearer_auth("dummy")
        .json(&json!({"read_only":"yes"}))
        .send()
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let unsupported = client
        .patch(&metadata_url)
        .bearer_auth("dummy")
        .json(&json!({"pinning":{"replicas":1}}))
        .send()
        .await
        .unwrap();
    assert_eq!(unsupported.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn delete_counts_and_filter_conditions_follow_live_rules() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/delete-rules");
    assert_eq!(
        post(
            &client,
            &url,
            json!({"upsert_rows":[{"id":2,"n":2},{"id":10,"n":10},{"id":1,"n":1}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    for body in [
        json!({"delete_by_filter":["n","Gt",0],"delete_condition":["n","Gt",5]}),
        json!({"patch_by_filter":{"filters":["n","Gt",0],"patch":{"n":7}},"patch_condition":["n","Gt",5]}),
    ] {
        assert_eq!(post(&client, &url, body).await.0, StatusCode::BAD_REQUEST);
    }
    let (status, result) = post(
        &client,
        &url,
        json!({"delete_by_filter":["n","Gt",0],"return_affected_ids":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["deleted_ids"], json!([1, 2, 10]));
    let (status, result) = post(
        &client,
        &url,
        json!({"deletes":[2,10],"return_affected_ids":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows_deleted"], 2);
    assert_eq!(result["deleted_ids"], json!([2, 10]));
    let (status, result) = post(
        &client,
        &url,
        json!({"deletes":[2,10],"delete_condition":["n","Gt",0],"return_affected_ids":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows_deleted"], 0);
    assert!(result.get("deleted_ids").is_none());
}

#[tokio::test]
async fn glob_filters_match_string_arrays_and_negate_missing_values() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/glob-arrays");
    assert_eq!(post(&client, &url, json!({
        "schema":{"id":"uint","tags":{"type":"[]string","glob":true}},
        "upsert_rows":[{"id":1,"tags":["Alpha","beta"]},{"id":2,"tags":["Gamma"]},{"id":3,"tags":[]},{"id":4}]
    })).await.0, StatusCode::OK);
    for (operator, pattern, expected) in [
        ("Glob", "A*", json!([1])),
        ("NotGlob", "A*", json!([2, 3, 4])),
        ("IGlob", "a*", json!([1])),
        ("NotIGlob", "a*", json!([2, 3, 4])),
        ("Glob", "*", json!([1, 2])),
        ("NotGlob", "*", json!([3, 4])),
    ] {
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":["id","asc"],"filters":["tags",operator,pattern],"limit":10}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let ids: Vec<Value> = result["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].clone())
            .collect();
        assert_eq!(json!(ids), expected, "{operator} {pattern}");
    }
}

#[tokio::test]
async fn for_each_unique_groups_distinct_array_values_and_missing_rows() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/unique-groups");
    assert_eq!(
        post(
            &client,
            &url,
            json!({
                "schema":{"id":"uint","tags":"[]string","g":"string","n":"int"},
                "upsert_rows":[
                    {"id":1,"tags":["a","b","a"],"g":"x","n":2},
                    {"id":2,"tags":["b","c"],"g":"x","n":3},
                    {"id":3,"tags":[],"g":"y","n":5},
                    {"id":4,"g":"y","n":7},
                    {"id":5,"tags":["a"],"n":11}
                ]
            })
        )
        .await
        .0,
        StatusCode::OK
    );
    let aggregate = json!({"count":["Count"],"sum":["Sum","n"]});
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"aggregate_by":aggregate,"group_by":[{"tag":["ForEachUnique","tags"]}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        result["aggregation_groups"],
        json!([
            {"tag":null,"count":1,"sum":7},
            {"tag":"a","count":2,"sum":13},
            {"tag":"b","count":2,"sum":5},
            {"tag":"c","count":1,"sum":3}
        ])
    );
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"aggregate_by":aggregate,"group_by":[{"tag":["ForEachUnique","tags"]},"g"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        result["aggregation_groups"],
        json!([
            {"tag":null,"g":"y","count":1,"sum":7},
            {"tag":"a","g":null,"count":1,"sum":11},
            {"tag":"a","g":"x","count":1,"sum":2},
            {"tag":"b","g":"x","count":2,"sum":5},
            {"tag":"c","g":"x","count":1,"sum":3}
        ])
    );
    for invalid in [
        json!([{"tag":["ForEachUnique","g"]}]),
        json!([{"tag":["ForEachUnique","missing"]}]),
        json!([{"tag":["ForEachUnique","tags"]},{"tag":["ForEachUnique","tags"]}]),
        json!([{"tag":["ForEachUnique","tags"]},{"other":["ForEachUnique","tags"]}]),
    ] {
        assert_eq!(
            post(
                &client,
                &format!("{url}/query"),
                json!({"aggregate_by":{"count":["Count"]},"group_by":invalid})
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
    }
}

#[tokio::test]
async fn null_writes_remove_attributes_from_returned_rows() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/null-writes");
    let query = json!({"rank_by":["id","asc"],"limit":10,"include_attributes":true});
    assert_eq!(
        post(
            &client,
            &url,
            json!({"upsert_rows":[{"id":1,"a":"x","n":5},{"id":2,"a":null,"n":null}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let (_, result) = post(&client, &format!("{url}/query"), query.clone()).await;
    assert_eq!(result["rows"], json!([{"id":1,"a":"x","n":5},{"id":2}]));
    assert_eq!(
        post(
            &client,
            &url,
            json!({"patch_rows":[{"id":1,"a":null,"n":null}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let (_, result) = post(&client, &format!("{url}/query"), query.clone()).await;
    assert_eq!(result["rows"], json!([{"id":1},{"id":2}]));
    assert_eq!(
        post(
            &client,
            &url,
            json!({"upsert_rows":[{"id":1,"a":null,"n":null}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let (_, result) = post(&client, &format!("{url}/query"), query).await;
    assert_eq!(result["rows"], json!([{"id":1},{"id":2}]));
}

#[tokio::test]
async fn null_bounds_follow_live_filter_ordering() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/null-bounds");
    assert_eq!(post(&client, &url, json!({"schema":{"id":"uint","n":"int"},"upsert_rows":[{"id":1,"n":1},{"id":2,"n":0},{"id":3}]})).await.0, StatusCode::OK);
    for (operator, expected) in [
        ("Lt", json!([])),
        ("Lte", json!([3])),
        ("Gt", json!([1, 2])),
        ("Gte", json!([1, 2, 3])),
    ] {
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":["id","asc"],"filters":["n",operator,null],"limit":10}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let ids: Vec<Value> = result["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].clone())
            .collect();
        assert_eq!(json!(ids), expected, "{operator}");
    }
    for operator in ["In", "NotIn"] {
        let (status, _) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":["id","asc"],"filters":["n",operator,[null]],"limit":10}),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }
}

#[tokio::test]
async fn id_and_attribute_name_boundaries_match_live() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/name-limits");
    let valid = json!({"upsert_rows":[{"id":"","a":1},{"id":"x".repeat(64),"a":2}]});
    assert_eq!(post(&client, &url, valid).await.0, StatusCode::OK);
    let mut long_ascii = json!({"id":1});
    long_ascii
        .as_object_mut()
        .unwrap()
        .insert("a".repeat(129), json!(1));
    let mut long_unicode = json!({"id":1});
    long_unicode
        .as_object_mut()
        .unwrap()
        .insert("é".repeat(65), json!(1));
    for row in [
        json!({"id":"x".repeat(65)}),
        json!({"id":1,"":1}),
        json!({"id":1,"$reserved":1}),
        long_ascii,
        long_unicode,
    ] {
        assert_eq!(
            post(&client, &url, json!({"upsert_rows":[row]})).await.0,
            StatusCode::BAD_REQUEST
        );
    }
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["id","asc"],"limit":10}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["rows"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn computed_bm25_uses_live_per_row_statistics() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/computed-bm25");
    assert_eq!(post(&client, &url, json!({
        "schema":{"id":"uint","t":{"type":"string","full_text_search":true}},
        "upsert_rows":[{"id":1,"t":"fugu"},{"id":2,"t":"fugu fugu"},{"id":3,"t":"other"},{"id":4,"t":"fugu whale sea"}]
    })).await.0, StatusCode::OK);
    let computed = json!({"s":["t","BM25","fugu"]});
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["id","asc"],"limit":10,"compute_attributes":computed}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    for (row, expected) in result["rows"].as_array().unwrap().iter().zip([
        std::f64::consts::LN_2,
        0.7438652,
        0.0,
        0.38123095,
    ]) {
        assert!((row["s"].as_f64().unwrap() - expected).abs() < 1e-5);
    }
    let (status, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["t","BM25","fugu"],"limit":10,"compute_attributes":computed}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(result["rows"][0]["$dist"], result["rows"][0]["s"]);
}

#[tokio::test]
async fn schema_object_updates_merge_options_and_shorthand_resets_them() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/schema-merge");
    let schema_url = url.replace("/v2/namespaces/", "/v1/namespaces/") + "/schema";
    assert_eq!(
        post(
            &client,
            &url,
            json!({"schema":{
        "id":"uint", "name":{"type":"string","full_text_search":{"k1":2.0,"b":0.2,"stemming":true},"glob":true},
        "label":"string"
    },"upsert_rows":[{"id":1,"name":"fugu","label":"fish"}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let (status, schema) = post(
        &client,
        &schema_url,
        json!({"name":{"type":"string","regex":true},"label":{"type":"string","regex":true}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(schema["name"]["glob"], true);
    assert_eq!(schema["name"]["regex"], true);
    assert_eq!(schema["name"]["filterable"], false);
    assert!(schema["name"]["full_text_search"].is_object());
    assert_eq!(schema["label"]["filterable"], true);
    let (status, schema) = post(
        &client,
        &schema_url,
        json!({"name":{"type":"string","full_text_search":{"k1":1.5}}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(schema["name"]["full_text_search"]["k1"], 1.5);
    assert_eq!(schema["name"]["full_text_search"]["b"], 0.2);
    assert_eq!(schema["name"]["full_text_search"]["stemming"], true);
    let (status, schema) = post(
        &client,
        &schema_url,
        json!({"name":{"type":"string","full_text_search":true}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(schema["name"]["full_text_search"]["k1"], 1.5);
    assert_eq!(schema["name"]["full_text_search"]["b"], 0.2);
    let (status, schema) = post(&client, &schema_url, json!({"name":"string"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(schema["name"]["filterable"], false);
    assert_eq!(schema["name"]["full_text_search"], Value::Null);
    assert!(schema["name"].get("glob").is_none());
    assert!(schema["name"].get("regex").is_none());
    assert_eq!(
        post(&client, &schema_url, json!({"name":{"regex":true}}))
            .await
            .0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn duplicate_explicit_ids_reject_the_entire_write() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/duplicate-ids");
    assert_eq!(
        post(
            &client,
            &url,
            json!({"upsert_rows":[{"id":1,"n":1},{"id":2,"n":2}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    for body in [
        json!({"upsert_rows":[{"id":3,"n":3},{"id":3,"n":4}]}),
        json!({"patch_rows":[{"id":1,"n":3},{"id":1,"n":4}]}),
        json!({"deletes":[1,1]}),
        json!({"deletes":[1],"upsert_rows":[{"id":1,"n":5}]}),
        json!({"patch_rows":[{"id":1,"n":5}],"upsert_rows":[{"id":1,"n":6}]}),
    ] {
        assert_eq!(post(&client, &url, body).await.0, StatusCode::BAD_REQUEST);
    }
    let (_, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["id","asc"],"limit":10,"include_attributes":true}),
    )
    .await;
    assert_eq!(result["rows"], json!([{"id":1,"n":1},{"id":2,"n":2}]));
    assert_eq!(
        post(
            &client,
            &url,
            json!({"delete_by_filter":["id","Eq",1],"upsert_rows":[{"id":1,"n":9}]})
        )
        .await
        .0,
        StatusCode::OK
    );
}

#[tokio::test]
async fn dense_vectors_cannot_be_patched_but_sparse_vectors_can() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/vector-patches");
    assert_eq!(post(&client, &url, json!({
        "distance_metric":"cosine_distance",
        "schema":{"id":"uint","vector":{"type":"[2]f32","ann":true},"s":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}}},
        "upsert_rows":[{"id":1,"vector":[1,0],"s":{"a":0.123456789}}]
    })).await.0, StatusCode::OK);
    assert_eq!(
        post(&client, &url, json!({"upsert_rows":[{"id":2,"s":{"a":1}}]}))
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
    let origin = base.trim_end_matches("/v2/namespaces");
    let schema: Value = client
        .get(format!("{origin}/v1/namespaces/vector-patches/schema"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(schema["vector"]["filterable"].is_null());
    let (_, initial) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["id","asc"],"limit":1,"include_attributes":true}),
    )
    .await;
    assert_eq!(initial["rows"][0]["s"]["a"], json!(0.12347412));
    assert_eq!(
        post(
            &client,
            &url,
            json!({"patch_rows":[{"id":1,"vector":[0,1]}]})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(
            &client,
            &url,
            json!({"patch_by_filter":{"filters":["id","Eq",1],"patch":{"vector":[0,1]}}})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post(
            &client,
            &url,
            json!({"patch_rows":[{"id":1,"s":{"a":0.654321}}]})
        )
        .await
        .0,
        StatusCode::OK
    );
    let (_, result) = post(
        &client,
        &format!("{url}/query"),
        json!({"rank_by":["id","asc"],"limit":10,"include_attributes":true}),
    )
    .await;
    assert_eq!(result["rows"][0]["vector"], json!([1.0, 0.0]));
    assert_eq!(result["rows"][0]["s"]["a"], json!(0.6542969));
}

#[tokio::test]
async fn vector_arrays_round_to_their_stored_element_width() {
    let base = server().await;
    let client = Client::new();
    for (kind, input, expected) in [
        ("f32", json!([1, 0.123456789]), json!([1.0, 0.12345679])),
        ("f16", json!([1, 0.123456789]), json!([1.0, 0.12347412])),
        ("i8", json!([1.0, 2.0]), json!([1, 2])),
    ] {
        let url = format!("{base}/vector-width-{kind}");
        let mut body = json!({"distance_metric":"cosine_distance","schema":{"id":"uint","vector":{"type":format!("[2]{kind}"),"ann":true}}});
        if kind == "f16" {
            body["upsert_columns"] = json!({"id":[1],"vector":[input]});
        } else {
            body["upsert_rows"] = json!([{"id":1,"vector":input}]);
        }
        assert_eq!(post(&client, &url, body).await.0, StatusCode::OK);
        let (status, result) = post(
            &client,
            &format!("{url}/query"),
            json!({"rank_by":["id","asc"],"limit":1,"include_attributes":["vector"]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result["rows"][0]["vector"], expected, "{kind}");
    }
}

#[tokio::test]
async fn aggregates_match_live_grouping_edges() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/aggregation-edges");
    let query_url = format!("{url}/query");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","g":"uint","h":"string","f":"float"},
            "upsert_rows":[
                {"id":1,"g":10,"h":"b","f":1.5},
                {"id":2,"g":2,"h":"c","f":2.5},
                {"id":3,"g":2,"h":"a","f":-4.0},
                {"id":4,"h":"z"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let aggregate = json!({"aggregate_by":{"count":["Count"],"sum":["Sum","f"]}});
    let (status, result) = post(
        &client,
        &query_url,
        json!({
            "aggregate_by":aggregate["aggregate_by"],"group_by":[]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["aggregations"], json!({"count":4,"sum":0.0}));

    for top_k in [json!(-1), json!("1"), json!(1.0), json!(true)] {
        let (status, _) = post(
            &client,
            &query_url,
            json!({"aggregate_by":{"count":["Count"]},"group_by":["g"],"top_k":top_k}),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }
    for group_by in [None, Some(json!([]))] {
        let mut query = json!({"aggregate_by":{"count":["Count"]},"top_k":0});
        if let Some(group_by) = group_by {
            query["group_by"] = group_by;
        }
        let (status, _) = post(&client, &query_url, query).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
    let (status, result) = post(
        &client,
        &query_url,
        json!({"aggregate_by":{"count":["Count"]},"top_k":null}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["aggregations"]["count"], 4);
    let (status, result) = post(
        &client,
        &query_url,
        json!({"aggregate_by":{"count":["Count"],"legacy_count":["Count","id"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["aggregations"], json!({"count":4,"legacy_count":4}));
    let (status, _) = post(
        &client,
        &query_url,
        json!({"aggregate_by":{"invalid":["Count","f"]}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, result) = post(
        &client,
        &query_url,
        json!({
            "aggregate_by":aggregate["aggregate_by"],"group_by":["g","h"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        result["aggregation_groups"],
        json!([
            {"g":null,"h":"z","count":1,"sum":0.0},
            {"g":2,"h":"a","count":1,"sum":-4.0},
            {"g":2,"h":"c","count":1,"sum":2.5},
            {"g":10,"h":"b","count":1,"sum":1.5}
        ])
    );

    for fields in [json!(["id"]), json!(["g", "g"])] {
        let (status, _) = post(
            &client,
            &query_url,
            json!({
                "aggregate_by":{"count":["Count"]},"group_by":fields
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn query_shape_errors_use_live_status_codes() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/query-shapes");
    let (status, _) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","group":"string","tags":"[]string"},
            "upsert_rows":[{"id":1,"group":"a","tags":["fish"]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let query_url = format!("{url}/query");
    for query in [
        json!({"rank_by":["id","asc"],"limit":"x"}),
        json!({"rank_by":["id","sideways"],"limit":1}),
        json!({"rank_by":["id","asc"],"filters":["id","Equals",1],"limit":1}),
        json!({"aggregate_by":{"count":["Avg","id"]}}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":["id"],"limit":1}),
        json!({"rank_by":"id","limit":1}),
        json!({"queries":{}}),
        json!({"rank_by":["id","asc"],"limit":1,"include_attributes":"id"}),
        json!({"rank_by":["id","asc"],"limit":1,"exclude_attributes":"id"}),
        json!({"rank_by":["id","asc"],"limit":1,"compute_attributes":[]}),
        json!({"rank_by":["id","asc"],"limit":1,"compute_attributes":{"x":["id","VectorDist"]}}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":"id"}),
        json!({"rank_by":["id","asc"],"limit":1,"consistency":"strong"}),
        json!({"rank_by":["id","asc"],"limit":1,"consistency":{"level":"invalid"}}),
        json!({"rank_by":["id","asc"],"limit":1,"vector_encoding":"invalid"}),
        json!({"rank_by":["id","asc"],"limit":1,"distance_metric":"invalid"}),
        json!({"rank_by":["id","asc"],"limit":1,"distance_metric":1}),
        json!({"rank_by":["id","asc"],"limit":1,"offset":-1}),
        json!({"rank_by":["id","asc"],"limit":1,"offset":"1"}),
        json!({"rank_by":["id","asc"],"limit":1,"offset":1.0}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["id","In",[1,"2"]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["group","In",["a",1]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","ContainsAny",[null]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","In",[["fish"]]]}),
        json!({"rank_by":["id","asc"],"limit":1.0}),
        json!({"rank_by":["id","asc"],"top_k":-1}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":null,"limit":1}}}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":["group"],"limit":null}},"include_attributes":["group"]}),
    ] {
        let (status, body) = post(&client, &query_url, query.clone()).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{query}: {body}");
        assert_eq!(body["status"], "error");
    }
    let subquery = json!({"rank_by":["id","asc"],"limit":1});
    for query in [
        json!({"rank_by":["id","asc"],"limit":0}),
        json!({"rank_by":["id","asc"],"top_k":0}),
        json!({"rank_by":["id","asc"],"limit":{"total":0}}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["id","Gt","1"]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["id","In",["1"]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","Contains",null]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","AnyGt",null]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","Gt",["a"]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","Gte",["fish"]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","Lt",["z"]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","Lte",["fish"]]}),
        json!({"rank_by":["id","asc"],"limit":null}),
        json!({"rank_by":["id","asc"],"top_k":null}),
        json!({"limit":1,"offset":0}),
        json!({"rank_by":["id","asc"],"limit":1,"exclude_attributes":["id"]}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":[],"limit":1}}}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":["group"],"limit":2}},"include_attributes":["group"]}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":["id"],"limit":1}},"include_attributes":true}),
    ] {
        assert_eq!(
            post(&client, &query_url, query).await.0,
            StatusCode::BAD_REQUEST
        );
    }
    for extra in [json!({"limit":0}), json!({"top_k":1})] {
        let mut query = json!({"queries":[subquery.clone()]});
        query
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let (status, body) = post(&client, &query_url, query).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["results"][0]["rows"][0]["id"], 1);
    }
    for query in [
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","In",["fish"]]}),
        json!({"rank_by":["id","asc"],"limit":1,"filters":["tags","NotIn",["other"]]}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"extra":true}}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":null}}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":["id"],"limit":1,"extra":true}},"include_attributes":["id"]}),
        json!({"rank_by":["id","asc"],"limit":{"total":1,"per":{"attributes":["group"],"limit":1}},"include_attributes":["group"]}),
    ] {
        let (status, body) = post(&client, &query_url, query).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["rows"][0]["id"], 1);
    }
    for count in [0, 17] {
        let (status, _) = post(
            &client,
            &query_url,
            json!({"queries":vec![subquery.clone(); count]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{count} subqueries");
    }
    let (status, result) = post(&client, &query_url, json!({"queries":vec![subquery; 16]})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["results"].as_array().unwrap().len(), 16);
    assert_eq!(
        post(
            &client,
            &query_url,
            json!({"rank_by":["id","asc"],"limit":1,"consistency":{"level":"strong","extra":1}})
        )
        .await
        .0,
        StatusCode::OK
    );
    for query in [
        json!({"rank_by":["id","asc"],"limit":10000,"offset":1}),
        json!({"rank_by":["id","asc"],"top_k":1,"offset":10000}),
        json!({"queries":[{"rank_by":["id","asc"],"limit":1},{"rank_by":["id","desc"],"limit":1}],"rerank_by":["RRF"],"limit":10000,"offset":1}),
    ] {
        assert_eq!(
            post(&client, &query_url, query).await.0,
            StatusCode::BAD_REQUEST
        );
    }
    assert_eq!(
        post(
            &client,
            &query_url,
            json!({"rank_by":["id","asc"],"limit":9999,"offset":1})
        )
        .await
        .0,
        StatusCode::OK
    );
    let malformed = client
        .post(&query_url)
        .bearer_auth("dummy")
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
    let body: Value = malformed.json().await.unwrap();
    assert_eq!(body["status"], "error");
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
    assert!((result["results"][0]["rows"][0]["$dist"].as_f64().unwrap() - 3.0 / 11.0).abs() < 1e-7);
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
    assert_eq!(
        body["schema"]["tag"],
        json!({"type":"string","filterable":true})
    );
    assert!(body["last_write_at"]
        .as_str()
        .unwrap()
        .ends_with(".000000000Z"));
    assert!(body["approx_logical_bytes"].as_u64().unwrap() > 0);
    assert!(body["created_at"].as_str().unwrap().ends_with('Z'));
    assert!(body["updated_at"].as_str().unwrap().ends_with('Z'));
    assert_eq!(body["encryption"], json!({"sse":true}));
    assert_eq!(body["index"]["status"], "up-to-date");
    let legacy: Value = client
        .get(format!("{origin}/v2/namespaces/inspection/metadata"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(legacy["created_at"], body["created_at"]);

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
    // The response is the live service's normalized schema view.
    assert_eq!(
        body["new"],
        json!({"type":"string","filterable":true,"full_text_search":null})
    );
    assert_eq!(
        body["id"],
        json!({"type":"uint","filterable":null,"full_text_search":null})
    );
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
    assert_eq!(recall["ground_truth"][1]["nearest_neighbors"][0]["id"], 1);

    let (status, explanation) = post(
        &client,
        &format!("{ns}/explain_query"),
        json!({"rank_by":["vector","ANN",[1,0]],"limit":1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        explanation["plan_text"],
        "MiniFugu exact unfiltered ranked scan of 2 rows"
    );

    let (status, multi_explanation) = post(
        &client,
        &format!("{ns}/explain_query"),
        json!({"queries":[{"rank_by":["vector","ANN",[1,0]],"limit":1},
                            {"rank_by":["vector","ANN",[0,1]],"limit":1}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        multi_explanation["plan_text"],
        "MiniFugu exact scan of 2 rows for 2 subqueries; local fusion"
    );

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
async fn recall_selects_an_ann_index_or_native_embedding() {
    let base = server().await;
    let client = Client::new();
    let origin = base.trim_end_matches("/v2/namespaces");
    let ns = format!("{base}/multiple-vectors");
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "schema":{"id":"uint","a_storage":{"type":"[][2]f32"},
                      "z_search":{"type":"[2]f32","ann":true}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"a_storage":[[0,1]],"z_search":[1,0]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, recall) = post(
        &client,
        &format!("{origin}/v1/namespaces/multiple-vectors/_debug/recall"),
        json!({"num":1,"top_k":1,"include_ground_truth":true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(recall["ground_truth"][0]["nearest_neighbors"][0]["id"], 1);

    let ns = format!("{base}/native-recall");
    let (status, _) = post(
        &client,
        &ns,
        json!({
            "schema":{"id":"uint","content":{"type":"string","embed":{"model":"test","dims":4}}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"content":"small fugu"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, recall) = post(
        &client,
        &format!("{origin}/v1/namespaces/native-recall/_debug/recall"),
        json!({"num":1,"top_k":1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "native recall: {recall}");
    assert_eq!(recall["avg_recall"], 1.0);
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
async fn narrow_vector_types_accept_float32_base64_and_return_native_width() {
    let base = server().await;
    let client = Client::new();
    let input = STANDARD.encode([1.0_f32.to_le_bytes(), 0.0_f32.to_le_bytes()].concat());
    for (kind, expected) in [("f16", vec![0, 0x3c, 0, 0]), ("i8", vec![1, 0])] {
        let ns = format!("{base}/encoded-{kind}");
        let (status, _) = post(
            &client,
            &ns,
            json!({
                "schema":{"id":"uint","vector":{"type":format!("[2]{kind}"),"ann":true}},
                "distance_metric":"cosine_distance",
                "upsert_rows":[{"id":1,"vector":input}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{kind} base64 input");
        let (status, response) = post(
            &client,
            &format!("{ns}/query"),
            json!({
                "rank_by":["vector","ANN",input],"limit":1,
                "include_attributes":["vector"],"vector_encoding":"base64"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{kind} base64 query");
        assert_eq!(response["rows"][0]["vector"], STANDARD.encode(expected));
    }
    let (status, _) = post(
        &client,
        &format!("{base}/encoded-i8"),
        json!({
            "upsert_rows":[{"id":2,"vector":[1.5,0]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
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
    let (status, override_result) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["vector","ANN",[1,1]],"limit":2,
            "distance_metric":"cosine_distance"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(override_result["rows"][0]["id"], 2);
    assert_eq!(override_result["rows"].as_array().unwrap().len(), 1);
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
        json!({"schema":{"title":{"type":"string","geo":true}}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("unsupported schema option geo"));
    for (index, schema) in [
        json!({"terms":{"type":"{}f16","sparse_knn":{"distance_metric":"cosine_distance"}}}),
        json!({"terms":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"},"filterable":true}}),
        json!({"blob":{"type":"bytes","filterable":true}}),
        json!({"vector":{"type":"[2]f32","ann":{"late_interaction":true}}}),
        json!({"tokens":{"type":"[][2]f32","ann":{"late_interaction":false}}}),
        json!({"tokens":{"type":"[][3073]f32"}}),
        json!({"title":{"type":"string","full_text_search":{"b":1.5}}}),
        json!({"title":{"type":"string","full_text_search":{"k1":0.0}}}),
    ]
    .into_iter()
    .enumerate()
    {
        let (status, body) = post(
            &client,
            &format!("{base}/invalid-schema-{index}"),
            json!({"schema":schema,"upsert_rows":[{"id":1}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "schema {schema}: {body}");
    }
    let sparse = format!("{base}/sparse-limits");
    let (status, _) = post(
        &client,
        &sparse,
        json!({
            "schema":{"terms":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}},"blob":"bytes"},
            "upsert_rows":[{"id":1,"terms":{"a":1.0}}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let oversized_query: serde_json::Map<String, Value> =
        (0..1025).map(|key| (key.to_string(), json!(1.0))).collect();
    let (status, _) = post(
        &client,
        &format!("{sparse}/query"),
        json!({"rank_by":["terms","SparseKNN",oversized_query],"limit":1}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    for rank in [json!(["terms", "asc"]), json!(["blob", "desc"])] {
        let (status, _) = post(
            &client,
            &format!("{sparse}/query"),
            json!({"rank_by":rank,"limit":1}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "rank {rank}");
    }
    // 8 MiB of float32 values in one multi-vector attribute is the documented ceiling.
    let tokens = format!("{base}/multivector-limits");
    let too_many_tokens = vec![vec![0.0_f32; 2]; 8 * 1024 * 1024 / 8 + 1];
    let (status, _) = post(
        &client,
        &tokens,
        json!({
            "schema":{"tokens":{"type":"[][2]f32"}},
            "upsert_rows":[{"id":1,"tokens":too_many_tokens}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let oversized_blob = STANDARD.encode(vec![0_u8; 8 * 1024 * 1024 + 1]);
    let (status, _) = post(
        &client,
        &sparse,
        json!({"upsert_rows":[{"id":2,"blob":oversized_blob}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = post(
        &client,
        &format!("{ns}/query"),
        json!({
            "rank_by":["id","asc"],"limit":1,"aggregate_by":{"count":["Count"]}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["error"].as_str().unwrap().contains("aggregation"));
    for (query, expected, expected_status) in [
        (
            json!({"rank_by":["id","asc"],"top_k":{"total":1,"per":{"attributes":["title"],"limit":1}},"include_attributes":["title"]}),
            "top_k must",
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"aggregate_by":{"count":["Count"]},"top_k":1}),
            "top_k requires a nonempty group_by",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"aggregate_by":{"title":["Count"]},"group_by":["title"],"top_k":1}),
            "conflicts with a group field",
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let (status, body) = post(&client, &format!("{ns}/query"), query).await;
        assert_eq!(status, expected_status);
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

#[tokio::test]
async fn schema_views_match_the_live_service() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/live_schema.json")).unwrap();
    let base = server().await;
    let client = Client::new();
    let origin = base.trim_end_matches("/v2/namespaces");
    let (status, body) = post(&client, &format!("{base}/shapes"), fixture["write"].clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let schema: Value = client
        .get(format!("{origin}/v1/namespaces/shapes/schema"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for (field, expected) in fixture["schema"].as_object().unwrap() {
        assert_eq!(&schema[field], expected, "schema view of {field}");
    }
    let metadata: Value = client
        .get(format!("{origin}/v1/namespaces/shapes/metadata"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for (field, expected) in fixture["metadata_schema"].as_object().unwrap() {
        assert_eq!(
            &metadata["schema"][field], expected,
            "metadata view of {field}"
        );
    }
}

#[tokio::test]
async fn embedded_attribute_schema_views_match_live_shapes() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/embedded-schema");
    let origin = base.trim_end_matches("/v2/namespaces");
    let (status, body) = post(&client, &url, json!({
        "schema":{"id":"uint","body":{"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":1536}}},
        "distance_metric":"cosine_distance",
        "upsert_rows":[{"id":1,"body":"red fish"}]
    })).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let schema: Value = client
        .get(format!("{origin}/v1/namespaces/embedded-schema/schema"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        schema["body"],
        json!({
            "type":"string","filterable":true,"full_text_search":null
        })
    );
    assert_eq!(
        schema["embed_body"],
        json!({
            "type":"[1536]f16","filterable":false,"full_text_search":null,"ann":true
        })
    );
    let metadata: Value = client
        .get(format!("{origin}/v1/namespaces/embedded-schema/metadata"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        metadata["schema"]["body"],
        json!({
            "type":"string","filterable":true,
            "embed":{"attribute":"embed_body","model":"openai/text-embedding-3-small"}
        })
    );
    assert_eq!(
        metadata["schema"]["embed_body"],
        json!({
            "type":"[1536]f16","filterable":false,
            "ann":{"distance_metric":"cosine_distance"}
        })
    );
}

#[tokio::test]
async fn integer_sum_keeps_values_beyond_float_precision() {
    let base = server().await;
    let client = Client::new();
    let url = format!("{base}/exact-sum");
    let (status, body) = post(
        &client,
        &url,
        json!({
            "schema":{"id":"uint","big":"uint"},
            "upsert_rows":[{"id":1,"big":9007199254740992_u64},{"id":2,"big":1}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = post(
        &client,
        &format!("{url}/query"),
        json!({
            "aggregate_by":{"sum":["Sum","big"]}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["aggregations"]["sum"], json!(9007199254740993_u64));
}

#[tokio::test]
async fn inferred_numeric_and_array_types_follow_live_rules() {
    let base = server().await;
    let client = Client::new();
    let origin = base.trim_end_matches("/v2/namespaces");
    let url = format!("{base}/inferred-types");
    let (status, body) = post(
        &client,
        &url,
        json!({
            "upsert_rows":[{"id":1,"count":3,"empty":[]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let schema: Value = client
        .get(format!("{origin}/v1/namespaces/inferred-types/schema"))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(schema["count"]["type"], "int");
    assert_eq!(schema["empty"]["type"], "[]unknown");

    let (status, body) = post(
        &client,
        &format!("{base}/inferred-decimal"),
        json!({
            "upsert_rows":[{"id":1,"fraction":1.5}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("signed 64-bit integer"));

    let (status, body) = post(
        &client,
        &format!("{base}/inferred-multivector"),
        json!({
            "upsert_rows":[{"id":1,"tokens":[[1.0,0.0],[0.0,1.0]]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let schema: Value = client
        .get(format!(
            "{origin}/v1/namespaces/inferred-multivector/schema"
        ))
        .bearer_auth("dummy")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(schema["tokens"]["type"], "[][2]f32");
}
