//! Optional seeded differential test against disposable live Turbopuffer namespaces.
//! Run with TURBOPUFFER_BASE_URL and TURBOPUFFER_API_KEY. It never reads existing data.

use reqwest::{Client, StatusCode};
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
    let response = client
        .post(url)
        .bearer_auth(token)
        .json(body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.json().await.unwrap();
    (status, body)
}

fn comparable(body: &Value) -> Value {
    if body.get("rows").is_some() {
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

fn scenario(seed: u64) -> (Value, Vec<Value>) {
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
            json!({
                "id":id,
                "title":titles[generator.next(titles.len())],
                "group":groups[generator.next(groups.len())],
                "score":generator.next(20) as i64,
                "weight":generator.next(20) as f64 / 2.0
            })
        })
        .collect::<Vec<_>>();
    let threshold = generator.next(15) as i64;
    let group = groups[generator.next(groups.len())];
    let term = ["fugu", "whale", "sea"][generator.next(3)];
    let write = json!({
        "schema":{
            "id":"uint","title":{"type":"string","full_text_search":true},
            "group":"string","score":"int","weight":"float"
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

    for seed in 1..=4 {
        let name = format!("minifugu-diff-{}", Uuid::new_v4().simple());
        let local_url = format!("{local_base}/v2/namespaces/{name}");
        let live_url = format!("{}/v2/namespaces/{name}", live_base.trim_end_matches('/'));
        let (write, queries) = scenario(0x5eed_2026_0929 + seed);
        let local_write = call(&client, "local", &local_url, &write).await;
        let live_write = call(&client, &live_token, &live_url, &write).await;
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
