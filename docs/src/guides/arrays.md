# Dense Matrices & Layout Rules

Julia arrays use **column-major** (Fortran) memory order, whereas `sciencekit_ffi` validates **row-major** (C-contiguous) descriptors (`SKFfiArray`). `ScienceKit.jl` handles this boundary transparently and reports whether a buffer was borrowed directly or transposed.

```mermaid
flowchart TD
    In["Input X :: AbstractMatrix{Float32 | Float64}"]:::in --> Check{"Memory already<br/>row-major?"}:::decision
    Check -->|"Matrix with rows==1 or cols==1<br/>or transpose(Matrix)"| Borrow["Zero-copy borrow<br/>path = :borrowed_contiguous<br/>copied = false"]:::zero
    Check -->|"Multi-column column-major Matrix<br/>or SubArray view"| Permute["permutedims(X) staging buffer<br/>path = :transposed_copy<br/>copied = true"]:::copy
    Borrow --> Preserve["GC.@preserve backing + shape + descriptor<br/>ccall into sciencekit_ffi"]:::ffi
    Permute --> Preserve

    classDef in fill:#4c1d95,stroke:#c4b5fd,color:#ffffff,stroke-width:2px
    classDef decision fill:#1e3a8a,stroke:#93c5fd,color:#ffffff,stroke-width:2px
    classDef zero fill:#065f46,stroke:#6ee7b7,color:#ffffff,stroke-width:2px
    classDef copy fill:#92400e,stroke:#fcd34d,color:#ffffff,stroke-width:2px
    classDef ffi fill:#0f766e,stroke:#5eead4,color:#ffffff,stroke-width:2px
```

## Inspecting conversion behavior

Use `inspect_matrix_conversion` and `inspect_vector_conversion` (tested in `test/arrays_tests.jl`) to audit memory behavior:

```julia
using ScienceKit

# Standard 2x3 column-major Matrix -> transposed into row-major staging buffer
X = [1.0 2.0 3.0; 4.0 5.0 6.0]
r1 = inspect_matrix_conversion(X)
# ConversionReport(:transposed_copy, (2, 3), Float64, true)

# Single-column 3x1 Matrix -> borrowed zero-copy!
X_col = reshape([10.0, 20.0, 30.0], 3, 1)
r2 = inspect_matrix_conversion(X_col)
# ConversionReport(:borrowed_contiguous, (3, 1), Float64, false)

# transpose(Matrix) where parent has shape (cols, rows) -> borrowed zero-copy!
X_parent = [1.0 4.0; 2.0 5.0; 3.0 6.0]
r3 = inspect_matrix_conversion(transpose(X_parent))
# ConversionReport(:borrowed_contiguous, (2, 3), Float64, false)
```

## Element type & shape validation

- **Supported element types:** `Float64` (`DTYPE_F64`) and `Float32` (`DTYPE_F32`). Passing `Matrix{Int64}` raises `ScienceKitError(STATUS_INVALID_ARGUMENT, ...)` suggesting `Float64.(X)` or `Float32.(X)`.
- **Non-empty dimensions:** Zero-row or zero-column matrices (`zeros(0, 2)`) and empty vectors (`Float64[]`) raise `ScienceKitError(STATUS_SHAPE, ...)`.
- **`GC.@preserve` discipline:** Every native call pins the backing `Array{T}`, the `Int64` shape vector, and the `Ref{FfiArrayDescriptor}` for the entire duration of the `ccall`.
