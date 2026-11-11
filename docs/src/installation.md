# Installation

## Requirements

- **Julia 1.10 LTS** or newer (`[compat] julia = "1.10"`)
- **Rust 1.85** or newer (to compile the native `sciencekit_ffi` shared library from [`neurono-ml/sciencekit`](https://github.com/neurono-ml/sciencekit))

## Why `sciencekit_ffi` is built from source

`sciencekit_ffi` is not yet published to `crates.io` or Julia's binary `Yggdrasil` registry. `ScienceKit.jl` therefore:

1. Vendors the authoritative C ABI header at `include/sciencekit.h` (verified against `neurono-ml/sciencekit` in CI via `scripts/sync-header.sh --check`).
2. Provides `scripts/build-native-library.sh` to compile `libsciencekit_ffi.{so,dylib,dll}` in release mode and install it into `lib/` where `ScienceKit.__init__()` discovers it automatically.

## Step-by-step setup

Clone both repositories side by side and build the native library:

```bash
git clone https://github.com/neurono-ml/sciencekit.git
git clone https://github.com/neurono-ml/sciencekit-jl.git
cd sciencekit-jl

# Compiles cargo build -p sciencekit_ffi --release and copies the library to ./lib/
scripts/build-native-library.sh ../sciencekit
```

Instantiate and verify the Julia package:

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Library discovery order

When `using ScienceKit` runs `__init__()`, `ScienceKit.jl` searches for the shared library in the following order:

1. `ENV["SCIENCEKIT_FFI_LIB"]` — explicit path to `libsciencekit_ffi.so` / `.dylib` / `.dll`.
2. `ENV["SCIENCEKIT_LIB_DIR"]` — directory containing the platform shared library.
3. `<package_root>/lib/<platform_library_name>` — populated by `scripts/build-native-library.sh`.
4. System dynamic loader paths (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `PATH`).

You can inspect the loaded library and ABI version at runtime:

```julia
using ScienceKit

@show library_path()
@show abi_version()   # v"1.0.0"
@show features()
```
