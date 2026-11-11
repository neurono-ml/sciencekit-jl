@testset "fit, predict, score, and transform" begin
    # Exact linear fixture: y = 1 + 2*x1 + 3*x2
    X = Float64[
        0.0 0.0;
        1.0 0.0;
        0.0 1.0;
        2.0 3.0;
    ]
    y = Float64[1.0, 3.0, 4.0, 14.0]

    builder = linear_regression_builder(; dtype=Float64, include_intercept=true, backend=:faer)
    model = fit(builder, X, y)
    @test !is_closed(model)

    preds = predict(model, X)
    @test preds isa Vector{Float64}
    @test length(preds) == 4
    @test isapprox(preds, y; atol=1e-9)

    r2 = score(model, X, y)
    @test r2 isa Float64
    @test isapprox(r2, 1.0; atol=1e-9)

    # Predict on new out-of-sample rows
    X_new = Float64[
        3.0 1.0;   # 1 + 6 + 3 = 10
        -1.0 2.0;  # 1 - 2 + 6 = 5
    ]
    @test isapprox(predict(model, X_new), [10.0, 5.0]; atol=1e-9)

    # Calling transform on a regressor raises STATUS_UNSUPPORTED
    unsupported_err = try
        transform(model, X)
        nothing
    catch e
        e
    end
    @test unsupported_err isa ScienceKitError
    @test unsupported_err.status == STATUS_UNSUPPORTED

    close!(model)
    close!(builder)

    # Float32 linear regression fixture
    X32 = Float32.(X)
    y32 = Float32.(y)
    builder32 = linear_regression_builder(; dtype=Float32, include_intercept=true)
    model32 = fit(builder32, X32, y32)
    preds32 = predict(model32, X32)
    @test preds32 isa Vector{Float32}
    @test isapprox(preds32, y32; atol=1f-4)
    @test isapprox(score(model32, X32, y32), 1.0; atol=1e-4)
    close!(model32)
    close!(builder32)

    # StandardScaler transform preserves (rows, cols) column-major geometry
    X_scale = Float64[
        1.0 10.0;
        2.0 20.0;
        3.0 30.0;
        4.0 40.0;
    ]
    scaler_builder = standard_scaler_builder(; dtype=Float64, with_mean=true, with_std=true)
    scaler_model = fit(scaler_builder, X_scale)
    Z = transform(scaler_model, X_scale)
    @test Z isa Matrix{Float64}
    @test size(Z) == (4, 2)
    # Column means are ~0 and population variances are ~1
    col_means = [sum(Z[:, j]) / size(Z, 1) for j in 1:size(Z, 2)]
    col_vars = [sum((Z[:, j] .- col_means[j]).^2) / size(Z, 1) for j in 1:size(Z, 2)]
    @test all(m -> abs(m) < 1e-12, col_means)
    @test all(v -> isapprox(v, 1.0; atol=1e-12), col_vars)
    # Both scaled columns should be identical because column 2 is 10 * column 1
    @test isapprox(Z[:, 1], Z[:, 2]; atol=1e-12)
    close!(scaler_model)
    close!(scaler_builder)

    # Table convenience fit_table
    tbl = (x1 = X[:, 1], x2 = X[:, 2], target = y, ignored = ["u", "v", "w", "z"])
    tbl_builder = linear_regression_builder()
    tbl_model = fit_table(tbl_builder, tbl; features=[:x1, :x2], target=:target)
    @test isapprox(predict(tbl_model, X), y; atol=1e-9)
    close!(tbl_model)
    close!(tbl_builder)
end

@testset "concurrency and large arrays" begin
    # Large synthetic linear dataset (4_000 x 8)
    n_rows, n_cols = 4_000, 8
    X_large = Matrix{Float64}(undef, n_rows, n_cols)
    weights = Float64[1.5, -2.0, 0.5, 3.0, -1.0, 0.25, 4.0, -0.75]
    for j in 1:n_cols, i in 1:n_rows
        X_large[i, j] = sin(0.01 * i * j) + 0.1 * cos(0.03 * i + j)
    end
    y_large = 2.5 .+ X_large * weights

    b_large = linear_regression_builder(; dtype=Float64)
    m_large = fit(b_large, X_large, y_large)
    @test isapprox(score(m_large, X_large, y_large), 1.0; atol=1e-9)

    # Concurrent predictions on a shared fitted model + concurrent independent fits
    tasks = [
        Threads.@spawn begin
            p = predict(m_large, X_large)
            local_b = linear_regression_builder()
            local_m = fit(local_b, X_large[1:200, :], y_large[1:200])
            local_r2 = score(local_m, X_large[1:200, :], y_large[1:200])
            close!(local_m)
            close!(local_b)
            return isapprox(p, y_large; atol=1e-8) && isapprox(local_r2, 1.0; atol=1e-8)
        end
        for _ in 1:8
    ]
    results = fetch.(tasks)
    @test all(results)

    close!(m_large)
    close!(b_large)
end
