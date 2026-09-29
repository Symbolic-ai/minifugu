# Changelog

Release Please maintains this file from conventional commits when a release pull request is merged.

## [0.2.0](https://github.com/Symbolic-ai/minifugu/compare/v0.1.1...v0.2.0) (2026-09-29)


### Features

* store generated embeddings in a named vector attribute ([#25](https://github.com/Symbolic-ai/minifugu/issues/25)) ([09a106b](https://github.com/Symbolic-ai/minifugu/commit/09a106b24afc58fca73e9703ebb1c25faee35381))
* support all hosted embedding models, embed dtype, encryption, and copy options ([#27](https://github.com/Symbolic-ai/minifugu/issues/27)) ([f402435](https://github.com/Symbolic-ai/minifugu/commit/f4024353ab215eb94e829d31cd487178d49fa08c))


### Bug Fixes

* match live no-op write errors and write response bodies ([#26](https://github.com/Symbolic-ai/minifugu/issues/26)) ([06d6ef0](https://github.com/Symbolic-ai/minifugu/commit/06d6ef0567e83a34ded21cf9833de396ea781025))

## [0.1.1](https://github.com/Symbolic-ai/minifugu/compare/v0.1.0...v0.1.1) (2026-09-29)


### Bug Fixes

* reject malformed arrays and require multi-vector upserts ([8355d12](https://github.com/Symbolic-ai/minifugu/commit/8355d1224ab5d0ea9e6815326b1513953b100b42))
* repair release workflow and document published crate ([3249958](https://github.com/Symbolic-ai/minifugu/commit/324995898d4112a044d1ee0416b92458a9dbc4f4))
* trigger required CI for release PRs with app token ([#24](https://github.com/Symbolic-ai/minifugu/issues/24)) ([542f990](https://github.com/Symbolic-ai/minifugu/commit/542f9905771bb654e4da62cdd233ba14ecbe1954))

## 0.1.0

Initial public release: a keyless local Turbopuffer API emulator with persistent namespaces, vector and BM25 search, filters, aggregations, highlighting, and optional embedding providers.
