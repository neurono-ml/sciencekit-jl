using Test
using ScienceKit

@testset "ScienceKit.jl" begin
    @testset "module load and ABI version" begin
        version = ScienceKit.abi_version()
        @test version.major == 1
        @test version.minor >= 0
        @test version.patch >= 0

        mask = ScienceKit.features()
        @test (mask & ScienceKit.FEATURE_LINEAR_REGRESSION) != 0
        @test (mask & ScienceKit.FEATURE_STANDARD_SCALER) != 0
        @test (mask & ScienceKit.FEATURE_MODEL_PERSISTENCE) != 0
        @test (mask & ScienceKit.FEATURE_F32) != 0
        @test (mask & ScienceKit.FEATURE_F64) != 0
        @test ScienceKit.has_feature(ScienceKit.FEATURE_LINEAR_REGRESSION)

        @test ScienceKit.status_name(ScienceKit.STATUS_OK) == "ok"
        @test ScienceKit.status_name(ScienceKit.STATUS_INVALID_HANDLE) == "invalid-handle"
        @test ScienceKit.status_name(ScienceKit.STATUS_SHAPE) == "shape"

        err = ScienceKitError(ScienceKit.STATUS_SHAPE, "row count mismatch")
        @test err isa Exception
        @test err.status == ScienceKit.STATUS_SHAPE
        @test err.message == "row count mismatch"
        @test occursin("shape", sprint(showerror, err))
        @test occursin("row count mismatch", sprint(showerror, err))
    end

    include("lifecycle_tests.jl")
    include("arrays_tests.jl")
    include("operations_tests.jl")
    include("persistence_and_errors_tests.jl")
end
