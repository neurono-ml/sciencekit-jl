@testset "builder and model lifecycle" begin
    GC.gc()
    baseline = ScienceKit.live_handle_count()

    builder = linear_regression_builder(; dtype=Float64, include_intercept=false, backend=:faer)
    @test !is_closed(builder)
    @test ScienceKit.live_handle_count() == baseline + 1

    set_include_intercept!(builder, true)
    set_backend!(builder, :matrix_multiply)
    set_backend!(builder, BACKEND_FAER)

    # Wrong-family setter raises STATUS_UNSUPPORTED without leaking or closing the builder
    err = try
        set_with_mean!(builder, false)
        nothing
    catch e
        e
    end
    @test err isa ScienceKitError
    @test err.status == STATUS_UNSUPPORTED
    @test !is_closed(builder)

    # Explicit close! is idempotent
    @test close!(builder) === nothing
    @test is_closed(builder)
    @test close!(builder) === nothing
    @test ScienceKit.live_handle_count() == baseline

    # Calling a setter on a closed builder raises STATUS_INVALID_HANDLE
    closed_err = try
        set_include_intercept!(builder, false)
        nothing
    catch e
        e
    end
    @test closed_err isa ScienceKitError
    @test closed_err.status == STATUS_INVALID_HANDLE

    # Invalid constructor configuration cleans up before rethrowing
    bad_cfg_err = try
        linear_regression_builder(; backend=:nonexistent_backend)
        nothing
    catch e
        e
    end
    @test bad_cfg_err isa ScienceKitError
    @test bad_cfg_err.status == STATUS_INVALID_ARGUMENT
    @test ScienceKit.live_handle_count() == baseline

    # GC finalizer releases unreachable builders without task-switch errors
    let
        for _ in 1:32
            standard_scaler_builder(; dtype=Float32, with_mean=true, with_std=false)
        end
    end
    GC.gc()
    GC.gc()
    @test ScienceKit.live_handle_count() == baseline
end
