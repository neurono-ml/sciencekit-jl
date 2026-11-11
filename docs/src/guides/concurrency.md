# Concurrency & Threading

Start Julia with multiple worker threads (`julia --threads=auto` or `JULIA_NUM_THREADS=4`) to call `ScienceKit.jl` concurrently across CPU cores.

## Threading rules

- **Fitted `Model` handles are immutable and `Send + Sync` in Rust:** Multiple Julia tasks (`Threads.@spawn` or `Threads.@threads`) may call `predict`, `transform`, `score`, and `save` concurrently on the same `Model` without external locking.
- **`Builder` handles are mutable configuration objects:** Use one `Builder` per task when fitting concurrently, or serialize configuration mutations on a shared builder.
- **Thread-local error isolation:** `sk_ffi_last_error` is stored in Rust `thread_local!` storage and copied into `ScienceKitError` immediately after a failed `ccall`, preventing cross-thread message clobbering.

## Concurrent inference & training example

Derived from `test/operations_tests.jl`:

```julia
using ScienceKit

X = Float64[0 0; 1 0; 0 1; 2 3]
y = Float64[1, 3, 4, 14]

shared_builder = linear_regression_builder()
shared_model = fit(shared_builder, X, y)

tasks = [
    Threads.@spawn begin
        # Concurrent inference on shared_model
        preds = predict(shared_model, X)

        # Independent fit on a task-local Builder
        local_b = linear_regression_builder()
        local_m = fit(local_b, X, y)
        r2 = score(local_m, X, y)
        close!(local_m)
        close!(local_b)

        isapprox(preds, y; atol=1e-9) && isapprox(r2, 1.0; atol=1e-9)
    end
    for _ in 1:8
]

@assert all(fetch.(tasks))

close!(shared_model)
close!(shared_builder)
```
