# Model Persistence

`save` and `load` call `sk_ffi_model_save` and `sk_ffi_model_load` across the C ABI boundary, writing and restoring models in `sciencekit`'s portable binary format. Models saved from Julia can be loaded from Rust, Go, or Lua bindings over the same `1.x` ABI.

## Saving and restoring a fitted model

Derived from `test/persistence_and_errors_tests.jl`:

```julia
using ScienceKit

X = Float64[0 0; 1 0; 0 1; 2 3]
y = Float64[1, 3, 4, 14]

mktempdir() do dir
    model_path = joinpath(dir, "linear_regression_f64.skbin")

    builder = linear_regression_builder(; dtype=Float64, include_intercept=true)
    model = fit(builder, X, y)
    save(model, model_path)

    # Original handles can be closed immediately; the file is self-contained
    close!(model)
    close!(builder)

    restored = load(model_path)
    @assert isapprox(predict(restored, X), y; atol=1e-12)
    @assert isapprox(score(restored, X, y), 1.0; atol=1e-12)
    close!(restored)
end
```

## Current ABI persistence scope

Per `include/sciencekit.h`, `sk_ffi_model_save` and `sk_ffi_model_load` currently support fitted `Float64` `LinearRegression` models:

- Attempting to `save` a `Float32` model or a fitted `StandardScaler` raises `ScienceKitError` with `STATUS_UNSUPPORTED`.
- Attempting to `load` a missing or unreadable file raises `ScienceKitError` with `STATUS_IO`.
- Passing an empty string `""` or a string containing an interior NUL byte `'\0'` raises `ScienceKitError` with `STATUS_INVALID_ARGUMENT` before crossing the C boundary.
