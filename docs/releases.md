# Releases

MiniFugu publishes its Rust crate to [crates.io](https://crates.io/crates/minifugu), which also builds its library documentation on [docs.rs](https://docs.rs/minifugu). Each version gets a GitHub Release with source archives and prebuilt `minifugu` binaries for Linux x86-64, macOS Intel and Apple Silicon, and Windows x86-64. Release assets include SHA-256 files.

## Normal release flow

1. Merge changes to `main` with [Conventional Commit](https://www.conventionalcommits.org/) squash titles: `fix:` for bug fixes, `feat:` for features, and `feat!:` or `BREAKING CHANGE:` for incompatible changes. Documentation and maintenance commits can use `docs:` and `chore:`.
2. The [release workflow](../.github/workflows/release.yml) uses Release Please to open or update a release PR with the next version in `Cargo.toml`, `Cargo.lock`, `.release-please-manifest.json`, and this changelog. GitHub does not trigger `pull_request` workflows for PRs created with `GITHUB_TOKEN`, so the release workflow dispatches CI on the release branch explicitly.
3. Review the version, changelog, CI, and automatic review. Merge the release PR only when these are clean. That merge creates the `vX.Y.Z` tag and GitHub Release. The same workflow then publishes the crate using a short-lived crates.io trusted-publishing token and uploads binaries. There is no long-lived crates.io token in GitHub Secrets.
4. Verify the crates.io version, docs.rs build, and GitHub Release assets. If publication or an asset upload fails after the release exists, rerun the release workflow with its `tag` input set to the existing `vX.Y.Z` tag. The crate publication step should be skipped if that exact version is already on crates.io; asset uploads overwrite that release's assets.

The version in the Git tag must match `Cargo.toml`. The release job verifies the package with `cargo package --locked` before publishing.

## First release setup

crates.io requires the first version of a new crate to be published with a crates.io API token before trusted publishing can be enabled. The `minifugu` name and `0.1.0` manifest are the bootstrap baseline. Once this workflow has merged and `main` passes CI:

1. Use a crates.io token with permission to publish a new crate. From a clean checkout of `main` at `Cargo.toml` version `0.1.0`, run `cargo package --locked`, then load the token into `CARGO_REGISTRY_TOKEN` without putting its value in shell history and run `cargo publish --locked`. Clear the variable afterwards. Never put the token in the repository.
2. Create the `v0.1.0` tag and GitHub Release from the exact published commit. Run the release workflow with `tag=v0.1.0` to attach the four binary archives. It detects that the crate version already exists and skips publication.
3. In the crate's crates.io **Settings → Trusted Publishing**, register GitHub owner `Symbolic-ai`, repository `minifugu`, workflow filename `release.yml`, and environment `release`. A crate owner must do this in the crates.io web UI. Confirm the release workflow can mint a trusted token on the next release, then revoke the first-publish token.

GitHub Actions is enabled to create pull requests in the repository's Actions settings. Release Please uses the built-in `GITHUB_TOKEN` with write access to Contents, Issues, Pull requests, and Actions for its release-management job. The crate-publishing job has `id-token: write` and read-only repository access. The `release` GitHub environment, restricted to `main`, is the trusted publisher identity. No stored GitHub or crates.io publishing token is needed after the first crates.io publication.
