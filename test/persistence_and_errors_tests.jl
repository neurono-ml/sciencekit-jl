@testset "persistence round-trip" begin
    X = Float64[
        0.0 0.0;
        1.0 0.0;
        0.0 1.0;
        2.0 3.0;
    ]
    y = Float64[1.0, 3.0, 4.0, 14.0]

    mktempdir() do dir
        model_path = joinpath(dir, "linear_regression_f64.skbin")

        builder = linear_regression_builder(; dtype=Float64, include_intercept=true)
        model = fit(builder, X, y)
        expected_preds = predict(model, X)
        @test save(model, model_path) === nothing
        @test isfile(model_path)

        # Close original handles before loading to prove independent state
        close!(model)
        close!(builder)

        restored = load(model_path)
        @test !is_closed(restored)
        @test isapprox(predict(restored, X), expected_preds; atol=1e-12)
        @test isapprox(score(restored, X, y), 1.0; atol=1e-12)
        close!(restored)
        @test is_closed(restored)

        # Saving a Float32 model is rejected with STATUS_UNSUPPORTED per ABI contract
        b32 = linear_regression_builder(; dtype=Float32)
        m32 = fit(b32, Float32.(X), Float32.(y))
        err32 = try
            save(m32, joinpath(dir, "f32.skbin"))
            nothing
        catch e
            e
        end
        @test err32 isa ScienceKitError
        @test err32.status == STATUS_UNSUPPORTED
        close!(m32)
        close!(b32)

        # Saving a StandardScaler model is rejected with STATUS_UNSUPPORTED per ABI contract
        bs = standard_scaler_builder()
        ms = fit(bs, X)
        errs = try
            save(ms, joinpath(dir, "scaler.skbin"))
            nothing
        catch e
            e
        end
        @test errs isa ScienceKitError
        @test errs.status == STATUS_UNSUPPORTED
        close!(ms)
        close!(bs)
    end
end

@testset "error mapping and repeated-failure leak check" begin
    GC.gc()
    GC.gc()
    baseline = ScienceKit.live_handle_count()

    X = Float64[0.0 0.0; 1.0 0.0; 0.0 1.0; 2.0 3.0]
    y = Float64[1.0, 3.0, 4.0, 14.0]
    X_wrong_cols = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    y_wrong_len = Float64[1.0, 2.0]

    builder = linear_regression_builder()
    model = fit(builder, X, y)
    scaler_builder = standard_scaler_builder()
    scaler_model = fit(scaler_builder, X)

    # Shape mismatch on fit -> STATUS_SHAPE
    e_fit_shape = try fit(builder, X, y_wrong_len); nothing catch e; e end
    @test e_fit_shape isa ScienceKitError
    @test e_fit_shape.status == STATUS_SHAPE
    @test !isempty(e_fit_shape.message)

    # Shape mismatch on predict (native ABI status) -> STATUS_SHAPE
    e_pred_shape = try predict(model, X_wrong_cols); nothing catch e; e end
    @test e_pred_shape isa ScienceKitError
    @test e_pred_shape.status == STATUS_SHAPE
    @test !isempty(e_pred_shape.message)

    # Unsupported predict on StandardScaler -> STATUS_UNSUPPORTED
    e_scaler_pred = try predict(scaler_model, X); nothing catch e; e end
    @test e_scaler_pred isa ScienceKitError
    @test e_scaler_pred.status == STATUS_UNSUPPORTED

    # Missing file on load -> STATUS_IO
    e_io = try load("/tmp/nonexistent_sciencekit_jl_model_$(rand(UInt64)).bin"); nothing catch e; e end
    @test e_io isa ScienceKitError
    @test e_io.status == STATUS_IO

    # Empty or NUL path -> STATUS_INVALID_ARGUMENT
    e_nul = try save(model, "bad\0path.bin"); nothing catch e; e end
    @test e_nul isa ScienceKitError
    @test e_nul.status == STATUS_INVALID_ARGUMENT

    e_empty_path = try load(""); nothing catch e; e end
    @test e_empty_path isa ScienceKitError
    @test e_empty_path.status == STATUS_INVALID_ARGUMENT

    # Closed model operations -> STATUS_INVALID_HANDLE
    closed_b = linear_regression_builder()
    closed_m = fit(closed_b, X, y)
    close!(closed_m)
    close!(closed_b)

    for op in (
        () -> fit(closed_b, X, y),
        () -> predict(closed_m, X),
        () -> transform(closed_m, X),
        () -> score(closed_m, X, y),
        () -> save(closed_m, "/tmp/should_not_write.bin"),
    )
        err = try op(); nothing catch e; e end
        @test err isa ScienceKitError
        @test err.status == STATUS_INVALID_HANDLE
    end

    # Repeated-failure leak check: 300 iterations across native & wrapper error paths
    active_before_loop = ScienceKit.live_handle_count()
    for _ in 1:300
        try fit(builder, X, y_wrong_len) catch end
        try fit(builder, Float64[NaN 1.0; 2.0 3.0], Float64[1.0, 2.0]) catch end
        try predict(model, X_wrong_cols) catch end
        try transform(model, X) catch end
        try predict(scaler_model, X) catch end
        try score(model, X_wrong_cols, Float64[1.0, 2.0]) catch end
        try load("/tmp/definitely_missing_sciencekit_jl_file.bin") catch end
        try predict(closed_m, X) catch end
    end
    GC.gc()
    GC.gc()
    @test ScienceKit.live_handle_count() == active_before_loop

    # Existing handles remain valid and clean up to baseline
    @test isapprox(predict(model, X), y; atol=1e-9)
    close!(scaler_model)
    close!(scaler_builder)
    close!(model)
    close!(builder)
    @test ScienceKit.live_handle_count() == baseline
end
