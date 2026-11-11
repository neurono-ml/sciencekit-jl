# Contributing

Contributions to `ScienceKit.jl` follow the same engineering discipline as the main [`neurono-ml/sciencekit`](https://github.com/neurono-ml/sciencekit) repository:

- **English everywhere:** code, docstrings, commits, issues, PRs, and documentation chapters are written in English.
- **Test-driven development (TDD):** write a failing `@testset` in `test/` first, implement the minimal change in `src/` to turn it green, and verify zero leaked handles via `live_handle_count()`.
- **Header synchronization:** never hand-edit `include/sciencekit.h`. Sync changes from `neurono-ml/sciencekit` using:

```bash
scripts/sync-header.sh /path/to/sciencekit
scripts/sync-header.sh --check /path/to/sciencekit
```

## Local verification gates

1. Build the shared library from `neurono-ml/sciencekit`:

```bash
scripts/build-native-library.sh /path/to/sciencekit
```

2. Run the full Julia test suite (single-threaded and multi-threaded):

```bash
JULIA_NUM_THREADS=4 julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

3. When updating documentation on `docs/documentations`, build the mdBook and regenerate `sitemap.xml` locally before pushing:

```bash
mdbook build docs
node docs/scripts/postbuild.mjs
```
