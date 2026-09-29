//! Run with TURBOPUFFER_BASE_URL and TURBOPUFFER_API_KEY to compare the same
//! disposable-namespace contract with the live service. No account is needed
//! for the ordinary test run.
use base64::{engine::general_purpose::STANDARD, Engine};
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
    let encoded = STANDARD.encode([1.0_f32.to_le_bytes(), 0.0_f32.to_le_bytes()].concat());
    let encoded_query = response(
        &client,
        token,
        &format!("{source}/query"),
        json!({
            "rank_by":["vector","ANN",encoded],"limit":1,
            "include_attributes":["vector"],"vector_encoding":"base64"
        }),
    )
    .await;
    let metadata_response = client
        .get(format!("{base}/v1/namespaces/{name}/metadata"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    let metadata_status = metadata_response.status();
    let metadata: Value = metadata_response.json().await.unwrap();
    let warm_response = client
        .get(format!("{base}/v1/namespaces/{name}/hint_cache_warm"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    let warm_status = warm_response.status();
    let warm: Value = warm_response.json().await.unwrap();
    let recall = response(
        &client,
        token,
        &format!("{base}/v1/namespaces/{name}/_debug/recall"),
        json!({"num":1,"top_k":1}),
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
    assert_eq!(
        encoded_query.0,
        StatusCode::OK,
        "base64 query: {:?}",
        encoded_query.1
    );
    assert_eq!(encoded_query.1["rows"][0]["id"], 1);
    assert!(encoded_query.1["rows"][0]["vector"].is_string());
    assert_eq!(metadata_status, StatusCode::OK);
    assert!(metadata["created_at"].is_string());
    assert_eq!(metadata["encryption"]["sse"], true);
    assert_eq!(warm_status, StatusCode::ACCEPTED);
    assert_eq!(warm["status"], "ACCEPTED");
    assert_eq!(recall.0, StatusCode::OK, "recall response: {:?}", recall.1);
    assert!(recall.1["avg_recall"].is_number());
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

async fn gap_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-gaps-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(&client, token, &url, json!({
        "schema":{
            "id":"uint",
            "sparse":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}},
            "name":{"type":"string","fuzzy":true},
            "blob":"bytes",
            "clicks":"uint",
            "delta":"int",
            "code":{"type":"string","regex":true},
            "title":{"type":"string","full_text_search":{"k1":2.0,"b":0.0,"k3":8.0}}
        },
        "upsert_rows":[
            {"id":1,"sparse":{"fish":1.0},"name":"Small Pufferfish","blob":"AP8=","clicks":100,"delta":-2,"code":"a-1","title":"orange orange fugu"},
            {"id":2,"sparse":{"fish":0.5},"name":"Blue whale","blob":"AAE=","clicks":10,"delta":5,"code":"b-2","title":"blue whale"},
            {"id":3,"name":"pufferfish are cute","clicks":0,"delta":0,"code":"c-3","title":"deep sea"}
        ]
    })).await;
    let sparse = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["sparse","SparseKNN",{"fish":1.0}],"limit":2
        }),
    )
    .await;
    let fuzzy = response(&client, token, &format!("{url}/query"), json!({
        "rank_by":["id","asc"],
        "filters":["name","Fuzzy","pufferfsh",{"max_edit_distance":[{"min_query_chars":6,"distance":1}],"case_sensitive":false}],
        "limit":10,"include_attributes":["blob"]
    })).await;
    let numeric = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["Saturate",["Attribute","clicks"],{"midpoint":100}],"limit":10
        }),
    )
    .await;
    let floor = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({"rank_by":["Max",[0.25,["title","BM25","fugu"]]],"limit":10}),
    )
    .await;
    let signed = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({"rank_by":["Attribute","delta"],"limit":10}),
    )
    .await;
    let clamped = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({"rank_by":["Max",[0,["Attribute","delta"]]],"limit":10}),
    )
    .await;
    let regex_eq = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({"filters":["code","Eq","a-1"],"limit":10}),
    )
    .await;
    let prefix = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["title","BM25","ora",{"last_as_prefix":true}],"limit":2
        }),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "gap write: {:?}", write.1);
    assert_eq!(sparse.0, StatusCode::OK, "sparse response: {:?}", sparse.1);
    assert_eq!(sparse.1["rows"][0]["id"], 1);
    assert_eq!(fuzzy.0, StatusCode::OK, "fuzzy response: {:?}", fuzzy.1);
    // The match may end before the end of the value; the whale row stays excluded.
    assert_eq!(ids(&fuzzy.1), [1, 3]);
    assert_eq!(fuzzy.1["rows"][0]["blob"], "AP8=");
    assert_eq!(
        numeric.0,
        StatusCode::OK,
        "numeric response: {:?}",
        numeric.1
    );
    // Attribute-derived scores keep rows that score zero.
    assert_eq!(ids(&numeric.1), [1, 2, 3]);
    assert_eq!(numeric.1["rows"][2]["$dist"], 0.0);
    assert_eq!(floor.0, StatusCode::OK, "floor response: {:?}", floor.1);
    // A scalar floor raises the score but does not add rows the text clause misses.
    assert_eq!(ids(&floor.1), [1]);
    assert_eq!(signed.0, StatusCode::BAD_REQUEST, "signed: {:?}", signed.1);
    assert_eq!(clamped.0, StatusCode::OK, "clamped: {:?}", clamped.1);
    assert_eq!(ids(&clamped.1), [2, 1, 3]);
    // regex makes filterable=false the default, as full_text_search does.
    assert_eq!(
        regex_eq.0,
        StatusCode::BAD_REQUEST,
        "regex Eq: {:?}",
        regex_eq.1
    );
    assert_eq!(prefix.0, StatusCode::OK, "prefix response: {:?}", prefix.1);
    assert_eq!(prefix.1["rows"][0]["id"], 1);
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
    gap_contract(&format!("http://{address}"), "dummy").await;
    ann_contract(&format!("http://{address}"), "dummy").await;
    null_sort_contract(&format!("http://{address}"), "dummy").await;
    regex_array_contract(&format!("http://{address}"), "dummy").await;
    inferred_vector_contract(&format!("http://{address}"), "dummy").await;
    embedded_write_contract(&format!("http://{address}"), "dummy").await;
    vector_lifecycle_contract(&format!("http://{address}"), "dummy").await;
    embedding_target_contract(&format!("http://{address}"), "dummy").await;
    embed_schema_contract(&format!("http://{address}"), "dummy").await;
    sort_validation_contract(&format!("http://{address}"), "dummy").await;
    query_embed_contract(&format!("http://{address}"), "dummy").await;
    array_shape_contract(&format!("http://{address}"), "dummy").await;
    multi_vector_upsert_contract(&format!("http://{address}"), "dummy").await;
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
    gap_contract(base.trim_end_matches('/'), &token).await;
    ann_contract(base.trim_end_matches('/'), &token).await;
    null_sort_contract(base.trim_end_matches('/'), &token).await;
    regex_array_contract(base.trim_end_matches('/'), &token).await;
    inferred_vector_contract(base.trim_end_matches('/'), &token).await;
    embedded_write_contract(base.trim_end_matches('/'), &token).await;
    vector_lifecycle_contract(base.trim_end_matches('/'), &token).await;
    embedding_target_contract(base.trim_end_matches('/'), &token).await;
    embed_schema_contract(base.trim_end_matches('/'), &token).await;
    sort_validation_contract(base.trim_end_matches('/'), &token).await;
    query_embed_contract(base.trim_end_matches('/'), &token).await;
    array_shape_contract(base.trim_end_matches('/'), &token).await;
    multi_vector_upsert_contract(base.trim_end_matches('/'), &token).await;
}

async fn array_shape_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-array-shapes-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let setup = response(
        &client,
        token,
        &url,
        json!({"schema":{"id":"uint","tags":"[]string"},"upsert_rows":[{"id":1,"tags":["base"]}]}),
    )
    .await;
    let mut results = Vec::new();
    for (write, expected) in [
        (
            json!({"upsert_rows":[{"id":2,"tags":["a",1]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"tags":["a",null]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"nested":[[1],["a"]]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_columns":{"id":[2],"tags":[["a",1]]}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"patch_rows":[{"id":1,"tags":["a",null]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"patch_columns":{"id":[1],"tags":[["a",1]]}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"patch_by_filter":{"filters":["id","Eq",1],"patch":{"tags":["a",1]}}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"vector":[1,"x"]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"vector":[1,null]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"schema":{"vector":{"type":"[2]f32","ann":true}},"distance_metric":"cosine_distance","upsert_rows":[{"id":2,"vector":[1,"x"]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"item":[{"a":1},"x"]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"item":[{"a":1},null]}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_rows":[{"id":2,"item":[{"a":1},{"a":2}]}]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"upsert_rows":[{"id":2,"tags":["a",1]}],"upsert_columns":{"id":[3],"tags":[["ok"]]}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        (
            json!({"upsert_columns":{"id":[2],"tags":[["one","two"]]}}),
            StatusCode::OK,
        ),
        (json!({"patch_rows":[{"id":1,"tags":[]}]}), StatusCode::OK),
        (
            json!({"patch_columns":{"id":[1],"tags":[["three","four"]]}}),
            StatusCode::OK,
        ),
        (
            json!({"patch_by_filter":{"filters":["id","Eq",1],"patch":{"tags":["five","six"]}}}),
            StatusCode::OK,
        ),
        (json!({"upsert_rows":[{"id":3,"tags":[]}]}), StatusCode::OK),
        (
            json!({"schema":{"nums":"[]float"},"upsert_rows":[{"id":5,"nums":[1,2.5]}]}),
            StatusCode::OK,
        ),
        (
            json!({"upsert_rows":[{"id":4,"nested":[[1],[2]]}]}),
            StatusCode::OK,
        ),
    ] {
        let (status, body) = response(&client, token, &url, write.clone()).await;
        results.push((write, expected, status, body));
    }
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(setup.0, StatusCode::OK, "array shape setup: {}", setup.1);
    for (write, expected, status, body) in results {
        assert_eq!(status, expected, "{write}: {body}");
    }
    assert_eq!(cleanup.status(), StatusCode::OK);
}

async fn multi_vector_upsert_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-multi-upserts-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let setup = response(
        &client,
        token,
        &url,
        json!({"schema":{"title":"string"},"upsert_rows":[{"id":1,"title":"base","mv":[[1.0],[2.0]]}]}),
    )
    .await;
    let mut results = Vec::new();
    for (write, expected) in [
        (
            json!({"upsert_rows":[{"id":2,"title":"no mv"}]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"upsert_rows":[{"id":2,"mv":[[3.0]]}]}),
            StatusCode::OK,
        ),
        (
            json!({"upsert_rows":[{"id":3,"mv":null}]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"upsert_rows":[{"id":3,"mv":[]}]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"upsert_columns":{"id":[4],"title":["no mv"]}}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"patch_rows":[{"id":1,"title":"patched"}]}),
            StatusCode::OK,
        ),
        (
            json!({"patch_columns":{"id":[1],"title":["patched again"]}}),
            StatusCode::OK,
        ),
    ] {
        let (status, body) = response(&client, token, &url, write.clone()).await;
        results.push((write, expected, status, body));
    }
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(setup.0, StatusCode::OK, "multi-vector setup: {}", setup.1);
    for (write, expected, status, body) in results {
        assert_eq!(status, expected, "{write}: {body}");
    }
    assert_eq!(cleanup.status(), StatusCode::OK);
}

async fn sort_validation_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-sort-validation-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","tags":"[]string","a0":"int","a1":"int","a2":"int","a3":"int","a4":"int","a5":"int","a6":"int","a7":"int","a8":"int"},
            "upsert_rows":[{"id":1,"tags":["fish"],"a0":0,"a1":1,"a2":2,"a3":3,"a4":4,"a5":5,"a6":6,"a7":7,"a8":8}]
        }),
    )
    .await;
    let order = |count| {
        (0..count)
            .map(|index| json!([format!("a{index}"), "asc"]))
            .collect::<Vec<_>>()
    };
    let cases = [
        (json!(["tags", "asc"]), StatusCode::BAD_REQUEST),
        (json!(["a0", "asc"]), StatusCode::OK),
        (json!(order(8)), StatusCode::OK),
        (json!(order(9)), StatusCode::BAD_REQUEST),
        (json!([]), StatusCode::BAD_REQUEST),
    ];
    let results: Result<Vec<_>, reqwest::Error> = async {
        let mut results = Vec::new();
        for (rank, expected) in cases {
            let reply = client
                .post(format!("{url}/query"))
                .bearer_auth(token)
                .json(&json!({"rank_by":rank,"limit":10}))
                .send()
                .await?;
            let status = reply.status();
            let body = reply.json::<Value>().await?;
            results.push((rank, expected, status, body));
        }
        Ok(results)
    }
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "sort write: {}", write.1);
    assert_eq!(cleanup.status(), StatusCode::OK);
    for (rank, expected, status, body) in results.expect("sort validation query failed") {
        assert_eq!(status, expected, "rank {rank}: {body}");
    }
}

async fn query_embed_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-query-embed-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","content":{"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":256}}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[
                {"id":1,"content":"pufferfish swim"},
                {"id":2,"content":"blue whale"}
            ]
        }),
    )
    .await;
    let cases = [
        (
            json!(["content", "ANN", ["Embed", "pufferfish"]]),
            StatusCode::OK,
            Some(1_u64),
        ),
        (
            json!(["embed_content", "ANN", ["Embed", "pufferfish", {"model":"openai/text-embedding-3-small"}]]),
            StatusCode::OK,
            Some(1),
        ),
        (
            json!(["content", "kNN", ["Embed", "pufferfish"]]),
            StatusCode::OK,
            Some(1),
        ),
        (
            json!(["content", "ANN", ["Embed", "pufferfish", {"model":"openai/text-embedding-3-large"}]]),
            StatusCode::OK,
            None,
        ),
        (
            json!(["embed_content", "ANN", ["Embed", "pufferfish"]]),
            StatusCode::BAD_REQUEST,
            None,
        ),
        (
            json!(["embed_content", "ANN", ["Embed", "pufferfish", {"model":null}]]),
            StatusCode::BAD_REQUEST,
            None,
        ),
        (
            json!(["embed_content", "ANN", ["Embed", "pufferfish", {"model":"openai/text-embedding-3-small","extra":true}]]),
            StatusCode::OK,
            Some(1),
        ),
        (
            json!(["embed_content", "ANN", ["Embed"]]),
            StatusCode::UNPROCESSABLE_ENTITY,
            None,
        ),
        (
            json!(["embed_content", "ANN", ["Embed", 12]]),
            StatusCode::UNPROCESSABLE_ENTITY,
            None,
        ),
    ];
    let results: Result<Vec<_>, reqwest::Error> = async {
        let mut results = Vec::new();
        for (rank, expected, winner) in cases {
            let mut query = json!({"rank_by":rank,"limit":2});
            if rank[1] == "kNN" {
                query["filters"] = json!(["id", "In", [1, 2]]);
            }
            let reply = client
                .post(format!("{url}/query"))
                .bearer_auth(token)
                .json(&query)
                .send()
                .await?;
            let status = reply.status();
            let body = reply.json::<Value>().await?;
            results.push((rank, expected, winner, status, body));
        }
        Ok(results)
    }
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "embed write: {}", write.1);
    assert_eq!(cleanup.status(), StatusCode::OK);
    for (rank, expected, winner, status, body) in results.expect("query embedding failed") {
        assert_eq!(status, expected, "rank {rank}: {body}");
        if let Some(winner) = winner {
            assert_eq!(body["rows"][0]["id"], winner, "rank {rank}: {body}");
        }
    }
}

async fn embed_schema_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-embed-schema-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let schema_url = format!("{base}/v1/namespaces/{name}/schema");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{
                "id":"uint",
                "short":{"type":"string","embed":"openai/text-embedding-3-small"},
                "object":{"type":"string","embed":{"model":"openai/text-embedding-3-small"}},
                "large":{"type":"string","embed":{"model":"openai/text-embedding-3-large"}},
                "narrow":{"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":256}}
            },
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"short":"red fish","object":"blue whale","large":"green turtle","narrow":"yellow crab"}]
        }),
    )
    .await;
    let schema: Result<(StatusCode, Value), reqwest::Error> = async {
        let response = client.get(&schema_url).bearer_auth(token).send().await?;
        let status = response.status();
        Ok((status, response.json().await?))
    }
    .await;
    let mut updates = Vec::new();
    for embed in [
        json!({"model":"openai/text-embedding-3-small"}),
        json!({"model":"openai/text-embedding-3-small","dims":null}),
    ] {
        let update = response(
            &client,
            token,
            &url,
            json!({
                "schema":{"narrow":{"type":"string","embed":embed}}
            }),
        )
        .await;
        let current: Value = client
            .get(&schema_url)
            .bearer_auth(token)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        updates.push((update, current));
    }
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "embed schema write: {}", write.1);
    assert_eq!(cleanup.status(), StatusCode::OK);
    let (status, schema) = schema.expect("embed schema request failed");
    assert_eq!(status, StatusCode::OK, "embed schema: {schema}");
    assert_eq!(schema["embed_short"]["type"], "[1536]f16");
    assert_eq!(schema["embed_object"]["type"], "[1536]f16");
    assert_eq!(schema["embed_large"]["type"], "[3072]f16");
    assert_eq!(schema["embed_narrow"]["type"], "[256]f16");
    for (update, schema) in updates {
        assert_eq!(update.0, StatusCode::OK, "embed update: {}", update.1);
        assert_eq!(schema["embed_narrow"]["type"], "[256]f16");
    }
}

async fn embedding_target_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-embed-target-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let schema_url = format!("{base}/v1/namespaces/{name}/schema");
    let metadata_url = format!("{base}/v1/namespaces/{name}/metadata");
    let setup = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","text":{"type":"string","embed":{
                "model":"openai/text-embedding-3-small","dims":256,"attribute":"vector"
            }}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"text":"pufferfish"},{"id":2,"text":"blue whale"}]
        }),
    )
    .await;
    let schema: Value = client
        .get(&schema_url)
        .bearer_auth(token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let metadata: Value = client
        .get(&metadata_url)
        .bearer_auth(token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let query = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["text","ANN",["Embed","pufferfish"]],"limit":2
        }),
    )
    .await;
    let mut vector = vec![0.0; 256];
    vector[0] = 1.0;
    let explicit = response(
        &client,
        token,
        &url,
        json!({
            "upsert_rows":[{"id":3,"vector":vector}]
        }),
    )
    .await;
    let projection = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["id","asc"],"limit":3,"include_attributes":["id","text","vector"]
        }),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(setup.0, StatusCode::OK, "{setup:?}");
    assert_eq!(schema["vector"]["type"], "[256]f16");
    assert!(schema.get("embed_text").is_none());
    assert_eq!(metadata["schema"]["text"]["embed"]["attribute"], "vector");
    assert_eq!(query.0, StatusCode::OK, "{query:?}");
    assert_eq!(query.1["rows"][0]["id"], 1);
    assert_eq!(explicit.0, StatusCode::OK, "{explicit:?}");
    assert_eq!(projection.0, StatusCode::OK, "{projection:?}");
    assert_eq!(ids(&projection.1), vec![1, 2, 3]);
    assert_eq!(
        projection.1["rows"][2]["vector"].as_array().unwrap().len(),
        256
    );
    assert_eq!(cleanup.status(), StatusCode::OK);
}

async fn embedded_write_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-embedded-write-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let setup = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","content":{"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":256}}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"content":"pufferfish"}]
        }),
    )
    .await;
    let mut vector = vec![0.0; 256];
    vector[0] = 1.0;
    vector[1] = 0.1234567;
    let mut out_of_range = vector.clone();
    out_of_range[0] = 1e6;
    let cases = [
        ("missing", json!({"id":2}), StatusCode::BAD_REQUEST),
        (
            "null",
            json!({"id":3,"content":null}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "empty",
            json!({"id":4,"content":""}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "vector",
            json!({"id":5,"embed_content":vector}),
            StatusCode::OK,
        ),
        (
            "both",
            json!({"id":6,"content":"fish","embed_content":vector}),
            StatusCode::OK,
        ),
        (
            "wrong_dims",
            json!({"id":7,"embed_content":[1.0,0.0]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "out_of_range",
            json!({"id":8,"embed_content":out_of_range}),
            StatusCode::BAD_REQUEST,
        ),
    ];
    let checks: Result<_, reqwest::Error> = async {
        let mut results = Vec::new();
        for (label, row, expected) in cases {
            let response = client
                .post(&url)
                .bearer_auth(token)
                .json(&json!({"upsert_rows":[row]}))
                .send()
                .await?;
            let status = response.status();
            let body = response.json::<Value>().await?;
            results.push((label, expected, status, body));
        }
        let response = client
            .post(format!("{url}/query"))
            .bearer_auth(token)
            .json(&json!({"rank_by":["id","asc"],"limit":10,"include_attributes":["id","content","embed_content"]}))
            .send()
            .await?;
        let status = response.status();
        let body = response.json::<Value>().await?;
        Ok::<_, reqwest::Error>((results, status, body))
    }
    .await;
    let invalid_source = response(
        &client,
        token,
        &url,
        json!({"schema":{"wrong":{"type":"int","embed":{"model":"openai/text-embedding-3-small","dims":256}}}}),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(setup.0, StatusCode::OK, "embedded write setup: {}", setup.1);
    assert_eq!(
        invalid_source.0,
        StatusCode::BAD_REQUEST,
        "{invalid_source:?}"
    );
    assert_eq!(cleanup.status(), StatusCode::OK);
    let (results, status, body) = checks.expect("embedded write request failed");
    for (label, expected, actual, reply) in results {
        assert_eq!(actual, expected, "{label}: {reply}");
    }
    assert_eq!(status, StatusCode::OK, "embedded query: {body}");
    assert_eq!(ids(&body), vec![1, 5, 6]);
    assert!(body["rows"][1].get("content").is_none());
    let mut stored_vector = vector;
    stored_vector[1] = half::f16::from_f32(stored_vector[1] as f32).to_f32() as f64;
    for row in &body["rows"].as_array().unwrap()[1..] {
        let values = row["embed_content"].as_array().unwrap();
        assert_eq!(values.len(), stored_vector.len());
        for (actual, expected) in values.iter().zip(&stored_vector) {
            assert_eq!(actual.as_f64().unwrap() as f32, *expected as f32);
        }
    }
    assert_eq!(body["rows"][2]["content"], "fish");
}

async fn inferred_vector_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-inferred-vector-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let schema_url = format!("{base}/v1/namespaces/{name}/schema");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "distance_metric":"cosine_distance",
            "upsert_rows":[
                {"id":1,"vector":"AAAAPwAAgD8="},
                {"id":2,"vector":[1.0,0.0]}
            ]
        }),
    )
    .await;
    let schema: Value = client
        .get(&schema_url)
        .bearer_auth(token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let query = response(
        &client,
        token,
        &format!("{url}/query"),
        json!({
            "rank_by":["vector","ANN",[1.0,0.0]],"limit":2
        }),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "{write:?}");
    assert_eq!(schema["vector"]["type"], "[2]f32");
    assert_eq!(schema["vector"]["ann"], true);
    assert_eq!(query.0, StatusCode::OK, "{query:?}");
    assert_eq!(ids(&query.1), vec![2, 1]);
    assert_eq!(cleanup.status(), StatusCode::OK);
}

async fn null_sort_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-null-sort-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","x":"int","y":"string"},
            "upsert_rows":[
                {"id":1,"x":3,"y":"b"},
                {"id":2,"y":"d"},
                {"id":3,"x":1,"y":"c"},
                {"id":4,"x":null,"y":"a"},
                {"id":5,"x":null,"y":null},
                {"id":6,"x":3,"y":null}
            ]
        }),
    )
    .await;
    let cases = [
        (json!(["x", "asc"]), vec![2, 4, 5, 3, 1, 6]),
        (json!(["x", "desc"]), vec![1, 6, 3, 2, 4, 5]),
        (json!([["x", "asc"], ["y", "asc"]]), vec![5, 4, 2, 3, 6, 1]),
        (json!([["x", "desc"], ["y", "asc"]]), vec![6, 1, 3, 5, 4, 2]),
    ];
    let results: Result<Vec<_>, reqwest::Error> = async {
        let mut results = Vec::new();
        for (rank, expected) in cases {
            let response = client
                .post(format!("{url}/query"))
                .bearer_auth(token)
                .json(&json!({"rank_by":rank,"limit":10}))
                .send()
                .await?;
            let status = response.status();
            let reply = response.json::<Value>().await?;
            results.push((rank, expected, status, reply));
        }
        Ok(results)
    }
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "null-sort write: {}", write.1);
    assert_eq!(cleanup.status(), StatusCode::OK);
    for (rank, expected, status, reply) in results.expect("null-sort query failed") {
        assert_eq!(status, StatusCode::OK, "rank {rank}: {reply}");
        assert_eq!(ids(&reply), expected, "rank {rank}");
    }
}

async fn regex_array_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-regex-array-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let write = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","tags":{"type":"[]string","regex":true,"filterable":true}},
            "upsert_rows":[
                {"id":1,"tags":["apple","fish"]},
                {"id":2,"tags":["whale"]},
                {"id":3,"tags":[]},
                {"id":4},
                {"id":5,"tags":[""]},
                {"id":6,"tags":null}
            ]
        }),
    )
    .await;
    let cases = [
        ("^fish$", vec![1]),
        ("apple.*fish", vec![]),
        ("^$", vec![5]),
        ("(?i)^WHALE$", vec![2]),
        ("^a", vec![1]),
    ];
    let mut results = Vec::new();
    for (pattern, expected) in cases {
        let (status, reply) = response(
            &client,
            token,
            &format!("{url}/query"),
            json!({"rank_by":["id","asc"],"filters":["tags","Regex",pattern],"limit":10}),
        )
        .await;
        results.push((pattern, expected, status, reply));
    }
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(write.0, StatusCode::OK, "regex-array write: {}", write.1);
    assert_eq!(cleanup.status(), StatusCode::OK);
    for (pattern, expected, status, reply) in results {
        assert_eq!(status, StatusCode::OK, "regex {pattern}: {reply}");
        assert_eq!(ids(&reply), expected, "regex {pattern}");
    }
}

async fn ann_contract(base: &str, token: &str) {
    let client = Client::new();
    let cases = [
        (json!({}), Some("cosine_distance"), StatusCode::OK),
        (
            json!({"distance_metric":"cosine_distance"}),
            Some("cosine_distance"),
            StatusCode::OK,
        ),
        (
            json!({"custom_option":"ignored"}),
            Some("cosine_distance"),
            StatusCode::OK,
        ),
        (
            json!({"distance_metric":null}),
            Some("cosine_distance"),
            StatusCode::OK,
        ),
        (json!(true), None, StatusCode::BAD_REQUEST),
        (
            json!({"distance_metric":"cosine_distance"}),
            None,
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"distance_metric":"euclidean_squared"}),
            Some("cosine_distance"),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"distance_metric":"invalid"}),
            Some("cosine_distance"),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ];
    for (ann, metric, expected) in cases {
        let name = format!("minifugu-ann-{}", Uuid::new_v4().simple());
        let url = format!("{base}/v2/namespaces/{name}");
        let mut body = json!({
            "schema":{"id":"uint","vector":{"type":"[2]f32","ann":ann}},
            "upsert_rows":[{"id":1,"vector":[1.0,0.0]}]
        });
        if let Some(metric) = metric {
            body["distance_metric"] = json!(metric);
        }
        let (status, reply) = response(&client, token, &url, body).await;
        let schema = if status == StatusCode::OK {
            let schema_url = format!("{base}/v1/namespaces/{name}/schema");
            let schema: Value = client
                .get(schema_url)
                .bearer_auth(token)
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            Some(schema)
        } else {
            None
        };
        let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
        assert_eq!(status, expected, "ANN write: {reply}");
        if let Some(schema) = schema {
            assert_eq!(cleanup.status(), StatusCode::OK);
            assert_eq!(schema["vector"]["ann"], true);
        }
    }
    for (ann, expected) in [
        (json!(true), StatusCode::BAD_REQUEST),
        (json!(false), StatusCode::OK),
        (json!({}), StatusCode::BAD_REQUEST),
    ] {
        let name = format!("minifugu-ann-scalar-{}", Uuid::new_v4().simple());
        let url = format!("{base}/v2/namespaces/{name}");
        let (status, reply) = response(
            &client,
            token,
            &url,
            json!({
                "schema":{"id":"uint","label":{"type":"string","ann":ann}},
                "upsert_rows":[{"id":1,"label":"one"}]
            }),
        )
        .await;
        let checks = if status == StatusCode::OK {
            let query_url = format!("{url}/query");
            let scalar_metric = response(
                &client,
                token,
                &query_url,
                json!({"rank_by":["id","asc"],"limit":1,"distance_metric":"cosine_distance"}),
            )
            .await;
            let aggregate_metric = response(
                &client,
                token,
                &query_url,
                json!({"aggregate_by":{"total":["Count"]},"distance_metric":"cosine_distance"}),
            )
            .await;
            Some((scalar_metric, aggregate_metric))
        } else {
            None
        };
        let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
        assert_eq!(status, expected, "scalar ANN write: {reply}");
        if let Some((scalar_metric, aggregate_metric)) = checks {
            assert_eq!(cleanup.status(), StatusCode::OK);
            assert_eq!(
                scalar_metric.0,
                StatusCode::BAD_REQUEST,
                "scalar metric: {}",
                scalar_metric.1
            );
            assert_eq!(
                aggregate_metric.0,
                StatusCode::UNPROCESSABLE_ENTITY,
                "aggregate metric: {}",
                aggregate_metric.1
            );
        }
    }
    for (metric, expected) in [
        (None, StatusCode::BAD_REQUEST),
        (Some("cosine_distance"), StatusCode::OK),
    ] {
        let name = format!("minifugu-embed-metric-{}", Uuid::new_v4().simple());
        let url = format!("{base}/v2/namespaces/{name}");
        let mut body = json!({
            "schema":{"id":"uint","content":{"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":1536}}},
            "upsert_rows":[{"id":1,"content":"tiny orange pufferfish"}]
        });
        if let Some(metric) = metric {
            body["distance_metric"] = json!(metric);
        }
        let (status, reply) = response(&client, token, &url, body).await;
        let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
        assert_eq!(status, expected, "embed metric: {reply}");
        if status == StatusCode::OK {
            assert_eq!(cleanup.status(), StatusCode::OK);
        }
    }
    for (metric, expected) in [
        ("cosine_distance", StatusCode::UNPROCESSABLE_ENTITY),
        ("dot_product", StatusCode::OK),
    ] {
        let name = format!("minifugu-sparse-metric-{}", Uuid::new_v4().simple());
        let url = format!("{base}/v2/namespaces/{name}");
        let (status, reply) = response(
            &client,
            token,
            &url,
            json!({
                "schema":{"id":"uint","s":{"type":"{}f16","sparse_knn":{"distance_metric":metric}}},
                "upsert_rows":[{"id":1,"s":{"fish":1.0}}]
            }),
        )
        .await;
        let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
        assert_eq!(status, expected, "sparse metric {metric}: {reply}");
        if status == StatusCode::OK {
            assert_eq!(cleanup.status(), StatusCode::OK);
        }
    }
    let name = format!("minifugu-ann-update-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let create = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"id":"uint","vector":{"type":"[2]f32","ann":true}},
            "distance_metric":"cosine_distance",
            "upsert_rows":[{"id":1,"vector":[1.0,0.0]}]
        }),
    )
    .await;
    let mut query_results = Vec::new();
    for (metric, expected) in [
        ("cosine_distance", StatusCode::OK),
        ("euclidean_squared", StatusCode::BAD_REQUEST),
    ] {
        let (status, reply) = response(
            &client,
            token,
            &format!("{url}/query"),
            json!({"rank_by":["vector","ANN",[1.0,0.0]],"distance_metric":metric,"limit":1}),
        )
        .await;
        query_results.push((metric, expected, status, reply));
    }
    let schema_url = format!("{base}/v1/namespaces/{name}/schema");
    let matching = response(
        &client,
        token,
        &schema_url,
        json!({"vector":{"type":"[2]f32","ann":{"distance_metric":"cosine_distance"}}}),
    )
    .await;
    let followup = response(
        &client,
        token,
        &url,
        json!({
            "schema":{"vector":{"type":"[2]f32","ann":{"distance_metric":"cosine_distance"}}},
            "upsert_rows":[{"id":2,"vector":[0.5,0.5]}]
        }),
    )
    .await;
    let mismatch = response(
        &client,
        token,
        &schema_url,
        json!({"vector":{"type":"[2]f32","ann":{"distance_metric":"euclidean_squared"}}}),
    )
    .await;
    let changed_metric = response(
        &client,
        token,
        &url,
        json!({"distance_metric":"euclidean_squared","upsert_rows":[{"id":2,"vector":[2.0,0.0]}]}),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(create.0, StatusCode::OK, "ANN update setup: {}", create.1);
    assert_eq!(cleanup.status(), StatusCode::OK);
    for (metric, expected, status, reply) in query_results {
        assert_eq!(status, expected, "query metric {metric}: {reply}");
    }
    assert_eq!(
        matching.0,
        StatusCode::OK,
        "matching ANN update: {}",
        matching.1
    );
    assert_eq!(matching.1["vector"]["ann"], true);
    assert_eq!(
        followup.0,
        StatusCode::OK,
        "ANN follow-up write: {}",
        followup.1
    );
    assert_eq!(
        mismatch.0,
        StatusCode::BAD_REQUEST,
        "ANN update: {}",
        mismatch.1
    );
    assert_eq!(
        changed_metric.0,
        StatusCode::BAD_REQUEST,
        "namespace metric change: {}",
        changed_metric.1
    );
}

async fn vector_lifecycle_contract(base: &str, token: &str) {
    let client = Client::new();
    let name = format!("minifugu-vector-lifecycle-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let create = response(
        &client,
        token,
        &url,
        json!({"schema":{"v1":{"type":"[2]f32","ann":true}},"distance_metric":"cosine_distance","upsert_rows":[{"id":1,"v1":[1,0]}]}),
    )
    .await;
    let add = response(
        &client,
        token,
        &url,
        json!({"schema":{"v2":{"type":"[2]f32","ann":true}},"upsert_rows":[{"id":1,"v1":[1,0],"v2":[0,1]}]}),
    )
    .await;
    let add_sparse = response(
        &client,
        token,
        &url,
        json!({"schema":{"s":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}}},"upsert_rows":[{"id":2,"v1":[1,0],"s":{"1":1.0}}]}),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(
        create.0,
        StatusCode::OK,
        "vector lifecycle create: {}",
        create.1
    );
    assert_eq!(
        add.0,
        StatusCode::BAD_REQUEST,
        "vector lifecycle add: {}",
        add.1
    );
    assert_eq!(add_sparse.0, StatusCode::OK, "sparse add: {}", add_sparse.1);
    assert_eq!(cleanup.status(), StatusCode::OK);

    let name = format!("minifugu-scalar-lifecycle-{}", Uuid::new_v4().simple());
    let url = format!("{base}/v2/namespaces/{name}");
    let scalar = response(
        &client,
        token,
        &url,
        json!({"upsert_rows":[{"id":1,"title":"a"}]}),
    )
    .await;
    let inferred_vector = response(
        &client,
        token,
        &url,
        json!({"upsert_rows":[{"id":2,"vector":[1,0]}]}),
    )
    .await;
    let inferred_multi = response(
        &client,
        token,
        &url,
        json!({"upsert_rows":[{"id":2,"mv":[[1,0],[0,1]]}]}),
    )
    .await;
    let added_embed = response(
        &client,
        token,
        &url,
        json!({"schema":{"content":{"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":1536}}}}),
    )
    .await;
    let cleanup = client.delete(&url).bearer_auth(token).send().await.unwrap();
    assert_eq!(scalar.0, StatusCode::OK, "scalar create: {}", scalar.1);
    assert_eq!(
        inferred_vector.0,
        StatusCode::BAD_REQUEST,
        "inferred vector: {}",
        inferred_vector.1
    );
    assert_eq!(
        inferred_multi.0,
        StatusCode::OK,
        "inferred non-indexed multi-vector: {}",
        inferred_multi.1
    );
    assert_eq!(
        added_embed.0,
        StatusCode::BAD_REQUEST,
        "embedded vector add: {}",
        added_embed.1
    );
    assert_eq!(cleanup.status(), StatusCode::OK);

    for (count, expected) in [(8, StatusCode::OK), (9, StatusCode::BAD_REQUEST)] {
        let name = format!("minifugu-vector-limit-{}", Uuid::new_v4().simple());
        let url = format!("{base}/v2/namespaces/{name}");
        let mut schema = (0..count)
            .map(|index| {
                let field = if index == 0 {
                    "vector".to_owned()
                } else {
                    format!("v{index}")
                };
                (field, json!({"type":"[2]f32","ann":true}))
            })
            .collect::<serde_json::Map<_, _>>();
        let mut row = serde_json::Map::from_iter([("id".into(), json!(1))]);
        for field in schema.keys() {
            row.insert(field.clone(), json!([1, 0]));
        }
        if count == 8 {
            schema.insert(
                "s".into(),
                json!({"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}}),
            );
            row.insert("s".into(), json!({"1":1.0}));
        }
        let result = response(
            &client,
            token,
            &url,
            json!({"schema":schema,"distance_metric":"cosine_distance","upsert_rows":[row]}),
        )
        .await;
        let _ = client.delete(&url).bearer_auth(token).send().await;
        assert_eq!(result.0, expected, "{count} vectors: {}", result.1);
    }

    for (fixed_count, expected) in [(7, StatusCode::OK), (8, StatusCode::BAD_REQUEST)] {
        let name = format!("minifugu-embedded-limit-{}", Uuid::new_v4().simple());
        let url = format!("{base}/v2/namespaces/{name}");
        let mut schema = serde_json::Map::new();
        let mut row = serde_json::Map::from_iter([
            ("id".into(), json!(1)),
            ("content".into(), json!("hello")),
            ("embed_content".into(), json!(vec![0.0; 1536])),
        ]);
        for index in 0..fixed_count {
            let field = if index == 0 {
                "vector".to_owned()
            } else {
                format!("v{index}")
            };
            schema.insert(field.clone(), json!({"type":"[2]f32","ann":true}));
            row.insert(field, json!([1, 0]));
        }
        schema.insert(
            "content".into(),
            json!({"type":"string","embed":{"model":"openai/text-embedding-3-small","dims":1536}}),
        );
        let result = response(
            &client,
            token,
            &url,
            json!({"schema":schema,"distance_metric":"cosine_distance","upsert_rows":[row]}),
        )
        .await;
        let _ = client.delete(&url).bearer_auth(token).send().await;
        assert_eq!(
            result.0, expected,
            "{fixed_count} fixed plus embedding: {}",
            result.1
        );
    }
}

fn ids(result: &Value) -> Vec<u64> {
    result["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_u64().unwrap())
        .collect()
}
