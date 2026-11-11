# API Reference

All symbols listed below are exported by `ScienceKit`.

## ABI & runtime metadata

| Function / Constant | Signature & Description |
| --- | --- |
| `abi_version()` | `() -> VersionNumber` — semantic version of the loaded `sciencekit_ffi` library (`v"1.0.0"`). |
| `features()` | `() -> UInt64` — bitmask of capabilities advertised by `sk_ffi_features`. |
| `has_feature(flag)` | `(flag::UInt64) -> Bool` — checks `FEATURE_LINEAR_REGRESSION`, `FEATURE_STANDARD_SCALER`, `FEATURE_MODEL_PERSISTENCE`, `FEATURE_F32`, `FEATURE_F64`. |
| `library_path()` | `() -> String` — resolved path or loader name of `libsciencekit_ffi`. |
| `status_name(s)` | `(s::Union{Status, Integer}) -> String` — canonical lowercase status label (`"ok"`, `"shape"`, ...). |

## Builders & configuration

| Function / Type | Signature & Description |
| --- | --- |
| `Builder` | `Builder(family::EstimatorFamily, dtype=Float64)` — mutable handle wrapping `SKFfiBuilder *`. |
| `linear_regression_builder` | `(; dtype=Float64, include_intercept=true, backend=:faer) -> Builder` |
| `standard_scaler_builder` | `(; dtype=Float64, with_mean=true, with_std=true, backend=:faer) -> Builder` |
| `set_include_intercept!` | `(builder::Builder, enabled::Bool) -> Builder` |
| `set_with_mean!` | `(builder::Builder, enabled::Bool) -> Builder` |
| `set_with_std!` | `(builder::Builder, enabled::Bool) -> Builder` |
| `set_backend!` | `(builder::Builder, backend::Union{BackendKind, Symbol, Integer}) -> Builder` (`:faer`, `:matrix_multiply`, `:ndarray_linalg`) |

## Fitting, inference, evaluation & persistence

| Function / Type | Signature & Description |
| --- | --- |
| `Model` | Opaque fitted or loaded model handle wrapping `SKFfiModel *`. |
| `fit` | `(builder::Builder, features, targets=nothing) -> Model` |
| `fit_table` | `(builder::Builder, table; features=nothing, target=nothing, missing_policy=:reject) -> Model` |
| `predict` | `(model::Model, features) -> Vector{Float64 \| Float32}` |
| `transform` | `(model::Model, features) -> Matrix{Float64 \| Float32}` |
| `score` | `(model::Model, features, targets) -> Float64` |
| `save` | `(model::Model, path::AbstractString) -> Nothing` |
| `load` | `(path::AbstractString) -> Model` |

## Conversion, lifecycle & errors

| Function / Type | Signature & Description |
| --- | --- |
| `extract_table` | `(table; features=nothing, target=nothing, dtype=Float64, missing_policy=:reject) -> (Matrix, Union{Vector, Nothing})` |
| `inspect_matrix_conversion` | `(X) -> ConversionReport` |
| `inspect_vector_conversion` | `(y) -> ConversionReport` |
| `ConversionReport` | Struct with fields `path::Symbol`, `shape::Tuple`, `dtype::DataType`, `copied::Bool`. |
| `close!` | `(handle::Union{Builder, Model}) -> Nothing` — idempotent deterministic release (also overloads `Base.close`). |
| `is_closed` | `(handle::Union{Builder, Model}) -> Bool` |
| `live_handle_count` | `() -> Int` — number of currently open native handles in this Julia process. |
| `ScienceKitError` | `Exception` carrying `status::Status`, `code::Int32`, and `message::String`. |
| `Status` | Enum: `STATUS_OK`, `STATUS_INVALID_HANDLE`, `STATUS_SHAPE`, `STATUS_INVALID_ARGUMENT`, `STATUS_UNSUPPORTED`, `STATUS_NOT_FITTED`, `STATUS_IO`, `STATUS_INTERNAL`. |
