# Quickstart

The example below mirrors `test/operations_tests.jl`: it fits an ordinary-least-squares linear regression model on an exact planar fixture $y = 1 + 2x_1 + 3x_2$, predicts on training and unseen samples, computes $R^2$, saves and reloads the fitted model, and releases every native handle deterministically.

```julia
using ScienceKit

# 4 samples x 2 features (Julia column-major Matrix{Float64})
X = Float64[
    0.0 0.0;
    1.0 0.0;
    0.0 1.0;
    2.0 3.0;
]
y = Float64[1.0, 3.0, 4.0, 14.0]

# 1. Configure and fit a LinearRegression builder
builder = linear_regression_builder(; dtype=Float64, include_intercept=true, backend=:faer)
model = fit(builder, X, y)

# 2. Predict and score
predictions = predict(model, X)          # [1.0, 3.0, 4.0, 14.0]
r2 = score(model, X, y)                  # 1.0

X_unseen = Float64[
    3.0  1.0;   # 1 + 2(3) + 3(1) = 10.0
   -1.0  2.0;   # 1 + 2(-1) + 3(2) = 5.0
]
unseen_predictions = predict(model, X_unseen)  # [10.0, 5.0]

# 3. Persist and reload
mktempdir() do dir
    path = joinpath(dir, "linear_model.skbin")
    save(model, path)

    restored = load(path)
    @assert isapprox(predict(restored, X_unseen), [10.0, 5.0]; atol=1e-9)
    close!(restored)
end

# 4. Deterministic cleanup (idempotent)
close!(model)
close!(builder)
```

## Scaling features with `StandardScaler`

```julia
using ScienceKit

X = Float64[
    1.0 10.0;
    2.0 20.0;
    3.0 30.0;
    4.0 40.0;
]

scaler_builder = standard_scaler_builder(; dtype=Float64, with_mean=true, with_std=true)
scaler = fit(scaler_builder, X)

Z = transform(scaler, X)   # 4x2 Matrix{Float64} with zero mean and unit population variance

close!(scaler)
close!(scaler_builder)
```
