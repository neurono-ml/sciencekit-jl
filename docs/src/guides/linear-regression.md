# Linear Regression

`linear_regression_builder` constructs a native `SKLinearRegression` builder over either `Float64` (default) or `Float32` precision and dispatches linear-algebra operations to the selected `sciencekit` math backend (`:faer`, `:matrix_multiply`, or `:ndarray_linalg`).

## Fitting, predicting, and scoring (`Float64`)

Derived from `test/operations_tests.jl`:

```julia
using ScienceKit

X = Float64[
    0.0 0.0;
    1.0 0.0;
    0.0 1.0;
    2.0 3.0;
]
y = Float64[1.0, 3.0, 4.0, 14.0]

builder = linear_regression_builder(;
    dtype=Float64,
    include_intercept=true,
    backend=:faer,
)
model = fit(builder, X, y)

preds = predict(model, X)
@assert preds isa Vector{Float64}
@assert isapprox(preds, y; atol=1e-9)
@assert isapprox(score(model, X, y), 1.0; atol=1e-9)

close!(model)
close!(builder)
```

## Single-precision (`Float32`) models

When `dtype=Float32` is selected on the builder, `fit` and `predict` validate that inputs are `AbstractMatrix{Float32}` and `AbstractVector{Float32}`, returning `Vector{Float32}` predictions while `score` returns `Float64`:

```julia
X32 = Float32[0 0; 1 0; 0 1; 2 3]
y32 = Float32[1, 3, 4, 14]

builder32 = linear_regression_builder(; dtype=Float32, include_intercept=true)
model32 = fit(builder32, X32, y32)

preds32 = predict(model32, X32)   # Vector{Float32}
r2 = score(model32, X32, y32)     # Float64

close!(model32)
close!(builder32)
```

## Reconfiguring a builder in place

Builders are mutable before and after `fit` calls (until `close!` is called):

```julia
builder = linear_regression_builder()
set_include_intercept!(builder, false)
set_backend!(builder, :matrix_multiply)
close!(builder)
```
