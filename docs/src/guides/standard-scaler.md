# Standard Scaler

`standard_scaler_builder` wraps `SKStandardScaler`, centering columns to zero mean (`with_mean=true`) and scaling to unit population standard deviation (`with_std=true`).

## Fitting and transforming a feature matrix

Derived from `test/operations_tests.jl`:

```julia
using ScienceKit

X = Float64[
    1.0 10.0;
    2.0 20.0;
    3.0 30.0;
    4.0 40.0;
]

builder = standard_scaler_builder(; dtype=Float64, with_mean=true, with_std=true)
scaler = fit(builder, X)

Z = transform(scaler, X)
@assert Z isa Matrix{Float64}
@assert size(Z) == (4, 2)
@assert isapprox(Z[:, 1], Z[:, 2]; atol=1e-12)

close!(scaler)
close!(builder)
```

## Disabling centering or variance scaling

Use keyword arguments on `standard_scaler_builder` or mutate an existing builder with `set_with_mean!` and `set_with_std!`:

```julia
builder = standard_scaler_builder(; with_mean=true, with_std=false)
set_with_std!(builder, true)
close!(builder)
```

Calling a regressor-only operation such as `predict(scaler, X)` on a fitted scaler raises `ScienceKitError` with `STATUS_UNSUPPORTED`.
