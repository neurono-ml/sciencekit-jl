# ScienceKit.jl

[![CI](https://github.com/neurono-ml/sciencekit-jl/actions/workflows/ci.yml/badge.svg)](https://github.com/neurono-ml/sciencekit-jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0d9488)](https://neurono-ml.github.io/sciencekit-jl/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-9558B2)](https://julialang.org/)

Idiomatic Julia bindings for the [`sciencekit`](https://github.com/neurono-ml/sciencekit) machine-learning library, calling the stable `sciencekit_ffi` C ABI directly through `ccall`.

Full documentation, guides, and API reference: **<https://neurono-ml.github.io/sciencekit-jl/>**

## Native ABI integration choice

`sciencekit_ffi` is not yet published to `crates.io` or `Yggdrasil`. `ScienceKit.jl` therefore vendors the authoritative C header at [`include/sciencekit.h`](include/sciencekit.h) (verified in CI with `scripts/sync-header.sh --check`) and loads the compiled `libsciencekit_ffi` shared library at module initialization (`__init__()`) via `Libdl.dlopen`.

Build and install the shared library from a checkout of [`neurono-ml/sciencekit`](https://github.com/neurono-ml/sciencekit):

```bash
scripts/build-native-library.sh /path/to/sciencekit
```

Or point `SCIENCEKIT_FFI_LIB` (full file path) or `SCIENCEKIT_LIB_DIR` (directory) to an existing build.

## Quickstart

```julia
using ScienceKit

X = Float64[
    0.0 0.0;
    1.0 0.0;
    0.0 1.0;
    2.0 3.0;
]
y = Float64[1.0, 3.0, 4.0, 14.0]

builder = linear_regression_builder(; dtype=Float64, include_intercept=true, backend=:faer)
model = fit(builder, X, y)

predictions = predict(model, X)
r2 = score(model, X, y)

close!(model)
close!(builder)
```

## Running the test suite

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
