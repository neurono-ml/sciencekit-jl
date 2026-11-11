# ScienceKit.jl

**ScienceKit.jl** provides idiomatic Julia bindings for [`sciencekit`](https://github.com/neurono-ml/sciencekit), a high-performance Rust machine-learning library reimplementing scikit-learn with strict numerical guarantees, panic containment, and zero-overhead native execution.

Rather than embedding a Julia runtime inside Rust (`jlrs`), `ScienceKit.jl` is a pure Julia package (`Project.toml` + `Libdl`) that loads the shared `sciencekit_ffi` library at module initialization (`__init__()`) and invokes its stable C ABI through native `ccall` function pointers.

```mermaid
flowchart LR
    A["Julia Caller<br/>Matrix / Vector / Table"]:::julia --> B["ScienceKit.jl<br/>Validation, Layout & GC.@preserve"]:::bridge
    B --> C["sciencekit_ffi (C ABI)<br/>Opaque Handles & Panic Guard"]:::abi
    C --> D["sciencekit Rust Core<br/>faer / matrixmultiply / rayon"]:::rust

    classDef julia fill:#6b21a8,stroke:#c084fc,color:#ffffff,stroke-width:2px
    classDef bridge fill:#0f766e,stroke:#5eead4,color:#ffffff,stroke-width:2px
    classDef abi fill:#1d4ed8,stroke:#93c5fd,color:#ffffff,stroke-width:2px
    classDef rust fill:#b45309,stroke:#fcd34d,color:#ffffff,stroke-width:2px
```

## Key capabilities

- **Direct `ccall` over the stable `1.x` C ABI:** `__init__()` resolves every symbol once via `Libdl.dlsym` and verifies ABI major-version compatibility (`abi_version().major == 1`) before user code runs.
- **Honest column-major ↔ row-major handling:** Julia stores matrices column-major, whereas `sciencekit_ffi` expects row-major buffers. `ScienceKit.jl` borrows 1-row, 1-column, and transposed `Matrix` wrappers zero-copy (`:borrowed_contiguous`) and transposes general `AbstractMatrix` inputs cleanly (`:transposed_copy`), reporting the exact path via `inspect_matrix_conversion`.
- **Columnar table ingestion:** `extract_table` and `fit_table` accept `NamedTuple` columns, row vectors of `NamedTuple`s, `Dict`s, `DataFrame`s, `CSV.File`, and `TypedTables.Table` with explicit `:reject`, `:nan`, and `:fill => value` missing-value policies.
- **Structured errors:** Every validation or native ABI failure raises `ScienceKitError(status, code, message)` carrying the typed `Status` enum and thread-local diagnostic message.
- **Deterministic & GC-safe handle lifecycle:** Builders and models use a non-yielding `Threads.SpinLock` lease counter so explicit `close!` is idempotent, concurrent calls never use-after-free, and Julia GC finalizers never trigger illegal task switches.
