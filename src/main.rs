use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    let address: SocketAddr = std::env::var("MINIFUGU_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:8787".to_owned())
        .parse()
        .expect("MINIFUGU_LISTEN must be a socket address");
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("failed to bind emulator listener");
    eprintln!("MiniFugu listening on http://{address}");
    let embedding = match std::env::var("MINIFUGU_EMBEDDING_PROVIDER").as_deref() {
        Ok("openai") => minifugu::EmbeddingMode::OpenAI {
            api_key: std::env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY is required in openai mode"),
            base_url: std::env::var("MINIFUGU_OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com".to_owned()),
        },
        Ok("deterministic") | Err(_) => minifugu::EmbeddingMode::Deterministic,
        Ok(_) => panic!("MINIFUGU_EMBEDDING_PROVIDER must be deterministic or openai"),
    };
    let router = match std::env::var("MINIFUGU_DATA_DIR") {
        Ok(directory) => {
            minifugu::router_with_data_dir(embedding, std::path::Path::new(&directory))
                .expect("failed to open MiniFugu data directory")
        }
        Err(_) => minifugu::router_with_mode(embedding),
    };
    axum::serve(listener, router).await.unwrap();
}
