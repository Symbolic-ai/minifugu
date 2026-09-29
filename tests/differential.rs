//! Optional seeded differential test against disposable live Turbopuffer namespaces.
//! Run with TURBOPUFFER_BASE_URL and TURBOPUFFER_API_KEY. It never reads existing data.

use reqwest::{Client, Method, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use uuid::Uuid;

struct Generator(u64);

impl Generator {
    fn next(&mut self, limit: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as usize) % limit
    }
}

async fn call(client: &Client, token: &str, url: &str, body: &Value) -> (StatusCode, Value) {
    request(client, Method::POST, token, url, Some(body)).await
}

async fn request(
    client: &Client,
    method: Method,
    token: &str,
    url: &str,
    body: Option<&Value>,
) -> (StatusCode, Value) {
    let mut request = client.request(method, url).bearer_auth(token);
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request.send().await.unwrap();
    let status = response.status();
    let body = response.json().await.unwrap();
    (status, body)
}

fn compare_write(
    failures: &mut Vec<String>,
    context: &str,
    local: &(StatusCode, Value),
    live: &(StatusCode, Value),
) {
    if local.0 != live.0 {
        failures.push(format!(
            "{context}: local status {}, live status {}",
            local.0, live.0
        ));
        return;
    }
    if !local.0.is_success() {
        return;
    }
    for key in [
        "status",
        "rows_affected",
        "rows_upserted",
        "rows_patched",
        "rows_deleted",
        "rows_remaining",
        "upserted_ids",
        "patched_ids",
        "deleted_ids",
    ] {
        if let Some(expected) = live.1.get(key) {
            if local.1.get(key) != Some(expected) {
                failures.push(format!(
                    "{context} {key}: local {:?}, live {expected}",
                    local.1.get(key)
                ));
            }
        }
    }
}

fn comparable(body: &Value) -> Value {
    if let Some(results) = body.get("results").and_then(Value::as_array) {
        json!({"results":results.iter().map(comparable).collect::<Vec<_>>()})
    } else if body.get("rows").is_some() {
        json!({"rows":body["rows"]})
    } else if body.get("aggregation_groups").is_some() {
        json!({"aggregation_groups":body["aggregation_groups"]})
    } else if body.get("aggregations").is_some() {
        json!({"aggregations":body["aggregations"]})
    } else {
        json!({"status":body["status"]})
    }
}

fn same_value(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => {
            if a.is_i64() && b.is_i64() {
                a == b
            } else {
                let (Some(a), Some(b)) = (a.as_f64(), b.as_f64()) else {
                    return false;
                };
                (a - b).abs() <= 1e-5_f64.max(a.abs() * 1e-5)
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_value(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| same_value(value, other)))
        }
        _ => left == right,
    }
}

fn scenario(seed: u64, metric: &str) -> (Value, Vec<Value>) {
    let mut generator = Generator(seed);
    let titles = [
        "tiny fugu",
        "blue whale",
        "orange fugu",
        "quiet sea",
        "fugu whale",
    ];
    let groups = ["a", "b", "c"];
    let rows = (1..=12)
        .map(|id| {
            let tags = if id == 7 {
                Value::Null
            } else if id % 4 == 0 {
                json!([])
            } else if id % 3 == 0 {
                json!(["fugu", "sea", "fugu"])
            } else {
                json!(["whale"])
            };
            json!({
                "id":id,
                "title":titles[generator.next(titles.len())],
                "group":groups[generator.next(groups.len())],
                "score":generator.next(20) as i64,
                "weight":generator.next(20) as f64 / 2.0,
                "tags":tags,
                "vector":[(1 + generator.next(9)) as f64 / 10.0,(1 + generator.next(9)) as f64 / 10.0],
                "sparse":{"fugu":0.123456789 + id as f64 / 100.0,"sea":0.3333333}
            })
        })
        .collect::<Vec<_>>();
    let threshold = generator.next(15) as i64;
    let group = groups[generator.next(groups.len())];
    let term = ["fugu", "whale", "sea"][generator.next(3)];
    let write = json!({
        "distance_metric":metric,
        "schema":{
            "id":"uint","title":{"type":"string","full_text_search":{"k1":1.8,"b":0.3}},
            "group":"string","score":"int","weight":"float",
            "tags":{"type":"[]string","glob":true,"filterable":true},
            "vector":{"type":"[2]f32","ann":true},
            "sparse":{"type":"{}f16","sparse_knn":{"distance_metric":"dot_product"}}
        },
        "upsert_rows":rows
    });
    let queries = vec![
        json!({"rank_by":["id","asc"],"limit":12}),
        json!({"rank_by":["id","desc"],"filters":["score","Gte",threshold],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["group","Eq",group],"limit":12}),
        json!({"rank_by":["title","BM25",term],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["title","ContainsAllTokens",term],"limit":12}),
        json!({"aggregate_by":{"count":["Count"],"sum":["Sum","score"],"weight":["Sum","weight"]}}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":["group","score"],"top_k":12}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":[]}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":["id"]}),
        json!({"rank_by":["id","asc"],"filters":["tags","NotIGlob","f*"],"limit":12}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":[{"tag":["ForEachUnique","tags"]}]}),
        json!({"rank_by":["id","asc"],"filters":["tags","Glob","f*"],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["tags","NotGlob","f*"],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["tags","Contains","fugu"],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["tags","NotContains","fugu"],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["tags","ContainsAny",["fugu","whale"]],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["tags","Eq",null],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","NotIn",[0,1,2]],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","Lt",threshold],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["group","NotEq",null],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","Lt",null],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","Lte",null],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","Gt",null],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","Gte",null],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","In",[null]],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["score","NotIn",[null]],"limit":12}),
        json!({"rank_by":[["group","asc"],["score","desc"]],"limit":12}),
        json!({"rank_by":["title","BM25","fugu"],"limit":{"total":12,"per":{"attributes":["group"],"limit":2}}}),
        json!({"rank_by":["id","asc"],"limit":5,"offset":2}),
        json!({"rank_by":["id","asc"],"limit":12,"compute_attributes":{"fugu_score":["title","BM25","fugu"]}}),
        json!({"aggregate_by":{"count":["Count"]},"group_by":[{"tag":["ForEachUnique","tags"]},"group"]}),
        json!({"rank_by":["vector","kNN",[0.2,0.7]],"filters":["id","Gte",1],"limit":5,"include_attributes":["vector"]}),
        json!({"rank_by":["sparse","SparseKNN",{"fugu":0.2,"sea":0.7}],"limit":8,"include_attributes":["sparse"]}),
        json!({"rank_by":["id","asc"],"filters":["And",[["score","Gte",threshold],["group","NotEq",null]]],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["Or",[["title","ContainsAnyToken","fugu"],["score","Lt",threshold]]],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["Not",["group","Eq",group]],"limit":12}),
        json!({"rank_by":["id","asc"],"filters":["title","ContainsTokenSequence","fugu whale"],"limit":12}),
        json!({"rank_by":["title","BM25","fu",{"last_as_prefix":true}],"limit":12}),
        json!({"rank_by":["Saturate",["Attribute","weight"],{"midpoint":3.0}],"limit":12}),
        json!({"rank_by":["Decay",["Dist",["Attribute","score"],7],{"midpoint":2}],"limit":12}),
        json!({"rank_by":["Max",[0,["Attribute","score"]]],"limit":12}),
        json!({"aggregate_by":{"count":["Count"],"sum":["Sum","score"]},"filters":["score","Gte",threshold],"group_by":["group"]}),
        json!({"rank_by":["id","asc"],"filters":["tags","ContainsAny",["whale","sea"]],"limit":12}),
    ];
    (write, queries)
}

#[tokio::test]
async fn generated_queries_match_live() {
    let (Ok(live_base), Ok(live_token)) = (
        std::env::var("TURBOPUFFER_BASE_URL"),
        std::env::var("TURBOPUFFER_API_KEY"),
    ) else {
        return;
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, minifugu::router()).await.unwrap() });
    let local_base = format!("http://{address}");
    let client = Client::new();
    let mut failures = Vec::new();

    for seed in 1..=6 {
        let name = format!("minifugu-diff-{}", Uuid::new_v4().simple());
        let local_url = format!("{local_base}/v2/namespaces/{name}");
        let live_url = format!("{}/v2/namespaces/{name}", live_base.trim_end_matches('/'));
        let metric = if seed <= 4 {
            "cosine_distance"
        } else {
            "euclidean_squared"
        };
        let (write, queries) = scenario(0x5eed_2026_0929 + seed, metric);
        let local_write = call(&client, "local", &local_url, &write).await;
        let live_write = call(&client, &live_token, &live_url, &write).await;
        compare_write(
            &mut failures,
            &format!("seed {seed} initial write"),
            &local_write,
            &live_write,
        );
        if local_write.0 != StatusCode::OK || live_write.0 != StatusCode::OK {
            failures.push(format!(
                "seed {seed} write: local {} live {}",
                local_write.0, live_write.0
            ));
        } else {
            for (index, query) in queries.iter().enumerate() {
                let local = call(&client, "local", &format!("{local_url}/query"), query).await;
                let live = call(&client, &live_token, &format!("{live_url}/query"), query).await;
                let local_value = comparable(&local.1);
                let live_value = comparable(&live.1);
                if local.0 != live.0 || !same_value(&local_value, &live_value) {
                    failures.push(format!(
                        "seed {seed} query {index} {query}: local {} {local_value}, live {} {live_value}",
                        local.0, live.0
                    ));
                }
            }
            let subqueries = json!([
                {"rank_by":["title","BM25","fugu"],"limit":8},
                {"rank_by":["vector","ANN",[0.2,0.7]],"limit":8}
            ]);
            for (index, query) in [
                json!({"queries":subqueries}),
                json!({"queries":subqueries,"rerank_by":["RRF",{"weights":[2,1],"rank_constant":10}],"limit":8}),
            ]
            .iter()
            .enumerate()
            {
                let local = call(&client, "local", &format!("{local_url}/query"), query).await;
                let live = call(&client, &live_token, &format!("{live_url}/query"), query).await;
                if local.0 != live.0 || !same_value(&comparable(&local.1), &comparable(&live.1)) {
                    failures.push(format!(
                        "seed {seed} multiquery {index}: local {} {}, live {} {}",
                        local.0,
                        comparable(&local.1),
                        live.0,
                        comparable(&live.1)
                    ));
                }
            }
            // Compare a sequence of writes and schema changes as well as their observable
            // row state. Each seed varies which conditional writes match.
            let changes = [
                json!({"patch_rows":[{"id":1,"title":"patched fugu"}],"return_affected_ids":true}),
                json!({"patch_rows":[{"id":2,"group":"z"}],"patch_condition":["score","Gt",10],"return_affected_ids":true}),
                json!({"upsert_rows":[{"id":13,"title":"new whale","group":"z","score":4,"weight":1.5}],"return_affected_ids":true}),
                json!({"delete_by_filter":["score","Lt",5],"return_affected_ids":true}),
                json!({"deletes":[3],"return_affected_ids":true}),
                json!({"patch_by_filter":{"filters":["group","Eq","z"],"patch":{"tags":null}},"return_affected_ids":true}),
                json!({"patch_columns":{"id":[1,2],"group":["col-1","col-2"]},"return_affected_ids":true}),
                json!({"upsert_columns":{"id":[14,15],"title":["column fugu","column whale"],"group":["c","d"],"score":[1,2],"weight":[1.0,2.0],"tags":[[],["fish"]]},"return_affected_ids":true}),
                json!({"upsert_rows":[{"id":16,"title":"first"},{"id":16,"title":"duplicate"}]}),
            ];
            let snapshot = json!({"rank_by":["id","asc"],"limit":20,"include_attributes":true});
            for (index, change) in changes.iter().enumerate() {
                let local = call(&client, "local", &local_url, change).await;
                let live = call(&client, &live_token, &live_url, change).await;
                compare_write(
                    &mut failures,
                    &format!("seed {seed} change {index}"),
                    &local,
                    &live,
                );
                let local = call(&client, "local", &format!("{local_url}/query"), &snapshot).await;
                let live = call(
                    &client,
                    &live_token,
                    &format!("{live_url}/query"),
                    &snapshot,
                )
                .await;
                if local.0 != live.0 || !same_value(&comparable(&local.1), &comparable(&live.1)) {
                    failures.push(format!(
                        "seed {seed} snapshot after change {index}: local {} {}, live {} {}",
                        local.0,
                        comparable(&local.1),
                        live.0,
                        comparable(&live.1)
                    ));
                }
            }
            let schema = json!({"title":{"type":"string","regex":true}});
            let local_schema_url =
                local_url.replace("/v2/namespaces/", "/v1/namespaces/") + "/schema";
            let live_schema_url =
                live_url.replace("/v2/namespaces/", "/v1/namespaces/") + "/schema";
            let local = call(&client, "local", &local_schema_url, &schema).await;
            let live = call(&client, &live_token, &live_schema_url, &schema).await;
            if local.0 != live.0 || !same_value(&local.1, &live.1) {
                failures.push(format!(
                    "seed {seed} schema update: local {} {}, live {} {}",
                    local.0, local.1, live.0, live.1
                ));
            }
            let tuning = json!({"title":{"type":"string","full_text_search":{"k1":1.5}}});
            let local = call(&client, "local", &local_schema_url, &tuning).await;
            let live = call(&client, &live_token, &live_schema_url, &tuning).await;
            if local.0 != live.0 || !same_value(&local.1, &live.1) {
                failures.push(format!(
                    "seed {seed} text tuning update: local {} {}, live {} {}",
                    local.0, local.1, live.0, live.1
                ));
            }
            let local_metadata_url =
                local_url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
            let live_metadata_url =
                live_url.replace("/v2/namespaces/", "/v1/namespaces/") + "/metadata";
            for enabled in [true, false] {
                let patch = json!({"read_only":enabled});
                let local = request(
                    &client,
                    Method::PATCH,
                    "local",
                    &local_metadata_url,
                    Some(&patch),
                )
                .await;
                let live = request(
                    &client,
                    Method::PATCH,
                    &live_token,
                    &live_metadata_url,
                    Some(&patch),
                )
                .await;
                if local.0 != live.0 || local.1.get("read_only") != live.1.get("read_only") {
                    failures.push(format!(
                        "seed {seed} read_only={enabled}: local {} {:?}, live {} {:?}",
                        local.0,
                        local.1.get("read_only"),
                        live.0,
                        live.1.get("read_only")
                    ));
                }
                if enabled {
                    let blocked = json!({"upsert_rows":[{"id":14,"title":"blocked"}]});
                    let local = call(&client, "local", &local_url, &blocked).await;
                    let live = call(&client, &live_token, &live_url, &blocked).await;
                    if local.0 != live.0 {
                        failures.push(format!(
                            "seed {seed} read-only write: local {}, live {}",
                            local.0, live.0
                        ));
                    }
                }
            }
        }
        let _ = client.delete(&local_url).bearer_auth("local").send().await;
        let _ = client
            .delete(&live_url)
            .bearer_auth(&live_token)
            .send()
            .await;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
