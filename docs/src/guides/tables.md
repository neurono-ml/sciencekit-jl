# Columnar Tables & Missingness

Julia's tabular ecosystem (`NamedTuple` column tables, `DataFrames.jl`, `CSV.File`, `Arrow.Table`, `TypedTables.Table`) represents datasets as named column vectors. `ScienceKit.jl` provides `extract_table` and `fit_table` so any column-accessible table works out of the box without adding heavy transitive dependencies to `Project.toml`.

## Selecting features and targets by column name

Derived from `test/arrays_tests.jl` and `test/operations_tests.jl`:

```julia
using ScienceKit

table = (
    x1 = [0.0, 1.0, 0.0, 2.0],
    x2 = [0.0, 0.0, 1.0, 3.0],
    target = [1.0, 3.0, 4.0, 14.0],
    sample_id = ["s1", "s2", "s3", "s4"],  # non-numeric column ignored when features are explicit
)

X, y = extract_table(table; features=[:x1, :x2], target=:target, dtype=Float64)

builder = linear_regression_builder()
model = fit_table(builder, table; features=[:x1, :x2], target=:target)
@assert isapprox(predict(model, X), y; atol=1e-9)

close!(model)
close!(builder)
```

## Missing-value policies

When a column has element type `Union{T, Missing}`:

| Policy | Behavior on `missing` |
| --- | --- |
| `:reject` (default) | Throws `ScienceKitError(STATUS_INVALID_ARGUMENT, ...)` naming the column and row index |
| `:nan` | Converts `missing` entries to `NaN` (`Float64` or `Float32`) |
| `:fill => value` | Replaces `missing` entries with `dtype(value)` |

```julia
tbl = (
    x1 = Union{Float64, Missing}[1.0, missing, 3.0],
    x2 = [1.0, 2.0, 3.0],
)

X_filled, _ = extract_table(tbl; features=[:x1, :x2], missing_policy=(:fill => 0.0))
@assert X_filled[2, 1] == 0.0
```

## Structured table validation errors

- **Unknown column name:** raises `KeyError(:column_name)`.
- **Uneven column lengths:** raises `DimensionMismatch`.
- **Duplicate feature column or non-numeric selected column:** raises `ScienceKitError` with `STATUS_INVALID_ARGUMENT`.
