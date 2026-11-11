# Errors and Status Codes

No Rust panic ever unwinds across `sciencekit_ffi`: every native entry point wraps execution in `std::panic::catch_unwind`, records a thread-local UTF-8 error message (`sk_ffi_last_error`), and returns a stable `SKFfiStatus` discriminant.

`ScienceKit.jl` copies that thread-local message immediately on the calling thread and throws a structured `ScienceKitError`.

## `ScienceKitError` structure

```julia
struct ScienceKitError <: Exception
    status::Status
    code::Int32
    message::String
end
```

## Stable `Status` codes

| Julia enum value | Code | `status_name` | Meaning |
| --- | ---: | --- | --- |
| `STATUS_OK` | `0` | `"ok"` | Operation succeeded |
| `STATUS_INVALID_HANDLE` | `1` | `"invalid-handle"` | Null, unknown, wrong-kind, or already-closed handle |
| `STATUS_SHAPE` | `2` | `"shape"` | Empty array, wrong rank, or mismatched row/feature counts |
| `STATUS_INVALID_ARGUMENT` | `3` | `"invalid-argument"` | Non-float dtype, dtype mismatch, `NaN`, invalid backend, or NUL path |
| `STATUS_UNSUPPORTED` | `4` | `"unsupported"` | Wrong-family setter/operation or unsupported persistence config |
| `STATUS_NOT_FITTED` | `5` | `"not-fitted"` | Estimator requires fitting before inference |
| `STATUS_IO` | `6` | `"io"` | Filesystem failure during library load, `save`, or `load` |
| `STATUS_INTERNAL` | `7` | `"internal"` | Internal error or caught Rust panic |

## Catching and inspecting errors

Derived from `test/persistence_and_errors_tests.jl`:

```julia
using ScienceKit

X = Float64[0 0; 1 0; 0 1; 2 3]
y = Float64[1, 3, 4, 14]

builder = linear_regression_builder()
model = fit(builder, X, y)

try
    # Passing 3 columns to a 2-feature model triggers STATUS_SHAPE
    predict(model, Float64[1 2 3; 4 5 6])
catch err
    if err isa ScienceKitError && err.status == STATUS_SHAPE
        @info "Caught shape mismatch" status=status_name(err.status) message=err.message
    else
        rethrow()
    end
finally
    close!(model)
    close!(builder)
end
```
