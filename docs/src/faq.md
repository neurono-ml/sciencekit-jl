# FAQ

## Why does `ScienceKit.jl` use `ccall` instead of `jlrs` or `CxxWrap.jl`?

Using `ccall` + `Libdl` against `sciencekit_ffi`'s pure C ABI keeps `ScienceKit.jl` a lightweight, pure-Julia package with zero external package dependencies (`Libdl` stdlib only), instant precompilation, and compatibility with standard Julia 1.10 LTS binaries without requiring C++ wrappers or pinning Rust to a newer MSRV.

## Why are general `Matrix{Float64}` inputs transposed before `ccall`?

Julia lays out 2-D arrays column-by-column, whereas `sciencekit`'s dense kernels and C ABI descriptors operate on row-major (`C`-contiguous) sample rows. Rather than hiding a layout mismatch or misinterpreting columns as rows, `ScienceKit.jl` transposes multi-column column-major matrices into a contiguous row-major staging buffer and reports `:transposed_copy` via `inspect_matrix_conversion`. If your data is already stored as `(features, samples)` in Julia, pass `transpose(raw_matrix)` to get a zero-copy `:borrowed_contiguous` call!

## Do I have to call `close!` manually if GC finalizers are registered?

GC finalizers ensure unreachable `Builder` and `Model` instances eventually free their Rust memory, even when exceptions occur. However, Julia's garbage collector tracks Julia heap pressure rather than Rust heap allocations — calling `close!(model)` explicitly as soon as a large model is no longer needed reclaims native memory deterministically.

## Can I load a model saved in Go, Lua, or Rust?

Yes. `save` and `load` delegate directly to `sk_ffi_model_save` and `sk_ffi_model_load` in `sciencekit_ffi`, so persisted `Float64` `LinearRegression` artifacts are byte-identical across every language binding over ABI `1.x`.
