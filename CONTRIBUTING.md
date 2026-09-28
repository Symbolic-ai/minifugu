# Contributing

Thanks for helping MiniFugu. Open an issue for a feature or behavior mismatch, ideally with a minimal synthetic HTTP request and the expected response. Do not include real API keys, customer data, or private application records.

## Local checks

Use the pinned Rust toolchain and run:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked
```

Add a focused HTTP test for new API behavior and update `docs/api-coverage.md`. Default tests must remain keyless and offline. Live compatibility tests require explicit environment variables and must use disposable synthetic namespaces. Keep errors explicit for unsupported fields.

By submitting a contribution, you agree to license it under the repository's MIT license. Logo artwork is excluded; do not add third-party artwork without documenting its rights.
