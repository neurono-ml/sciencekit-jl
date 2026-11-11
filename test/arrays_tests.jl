@testset "array and table conversion validation" begin
    X_f64 = [1.0 2.0 3.0; 4.0 5.0 6.0]
    report_2d = inspect_matrix_conversion(X_f64)
    @test report_2d.shape == (2, 3)
    @test report_2d.dtype == Float64
    @test report_2d.path == :transposed_copy
    @test report_2d.copied == true

    # Single-column contiguous Matrix{Float64} has identical memory order in column-major and row-major
    X_1col = reshape([10.0, 20.0, 30.0], 3, 1)
    report_1col = inspect_matrix_conversion(X_1col)
    @test report_1col.shape == (3, 1)
    @test report_1col.path == :borrowed_contiguous
    @test report_1col.copied == false

    # SubArray view is accepted and reported honestly
    X_view = @view X_f64[:, 1:2]
    report_view = inspect_matrix_conversion(X_view)
    @test report_view.shape == (2, 2)
    @test report_view.dtype == Float64
    @test report_view.copied == true

    # Adjoint of a column-major (3x2) matrix is already row-major in memory -> borrowed_contiguous!
    X_col_source = [1.0 4.0; 2.0 5.0; 3.0 6.0] # 3x2, adjoint is 2x3 [1 2 3; 4 5 6]
    X_adj = transpose(X_col_source)
    report_adj = inspect_matrix_conversion(X_adj)
    @test report_adj.shape == (2, 3)
    @test report_adj.path == :borrowed_contiguous
    @test report_adj.copied == false

    # Integer matrix rejected with STATUS_INVALID_ARGUMENT and cast hint
    int_err = try
        inspect_matrix_conversion([1 2; 3 4])
        nothing
    catch e
        e
    end
    @test int_err isa ScienceKitError
    @test int_err.status == STATUS_INVALID_ARGUMENT
    @test occursin("Float64", int_err.message)

    # Empty dimensions rejected with STATUS_SHAPE
    empty_err = try
        inspect_matrix_conversion(zeros(Float64, 0, 2))
        nothing
    catch e
        e
    end
    @test empty_err isa ScienceKitError
    @test empty_err.status == STATUS_SHAPE

    empty_vec_err = try
        inspect_vector_conversion(Float64[])
        nothing
    catch e
        e
    end
    @test empty_vec_err isa ScienceKitError
    @test empty_vec_err.status == STATUS_SHAPE

    # Columnar NamedTuple table extraction (Tables.jl-compatible column interface)
    tbl = (
        x1 = [0.0, 1.0, 0.0, 2.0],
        x2 = [0.0, 0.0, 1.0, 3.0],
        y = [1.0, 3.0, 4.0, 14.0],
        label = ["a", "b", "c", "d"],
    )
    X_tbl, y_tbl = extract_table(tbl; features=[:x1, :x2], target=:y)
    @test size(X_tbl) == (4, 2)
    @test X_tbl[:, 1] == [0.0, 1.0, 0.0, 2.0]
    @test X_tbl[:, 2] == [0.0, 0.0, 1.0, 3.0]
    @test y_tbl == [1.0, 3.0, 4.0, 14.0]

    # Row-table Vector{NamedTuple} and Dict table containers
    row_tbl = [(x1 = 0.0, x2 = 0.0, y = 1.0), (x1 = 1.0, x2 = 0.0, y = 3.0)]
    X_rows, y_rows = extract_table(row_tbl; features=[:x1, :x2], target=:y)
    @test size(X_rows) == (2, 2) && y_rows == [1.0, 3.0]

    dict_tbl = Dict(:x1 => [0.0, 2.0], :x2 => [1.0, 3.0], :y => [4.0, 14.0])
    X_dict, y_dict = extract_table(dict_tbl; features=[:x1, :x2], target=:y)
    @test size(X_dict) == (2, 2) && y_dict == [4.0, 14.0]

    # Unknown column raises KeyError
    @test_throws KeyError extract_table(tbl; features=[:x1, :missing_col])

    # Uneven column lengths raise DimensionMismatch
    bad_tbl = (x1 = [1.0, 2.0], x2 = [1.0, 2.0, 3.0])
    @test_throws DimensionMismatch extract_table(bad_tbl; features=[:x1, :x2])

    # Missing values under :reject raise ScienceKitError naming the column
    missing_tbl = (x1 = Union{Float64, Missing}[1.0, missing, 3.0], x2 = [1.0, 2.0, 3.0])
    miss_err = try
        extract_table(missing_tbl; features=[:x1, :x2], missing_policy=:reject)
        nothing
    catch e
        e
    end
    @test miss_err isa ScienceKitError
    @test miss_err.status == STATUS_INVALID_ARGUMENT
    @test occursin("x1", miss_err.message)

    # Missing values under :nan and (:fill => 0.0)
    X_nan, _ = extract_table(missing_tbl; features=[:x1, :x2], missing_policy=:nan)
    @test isnan(X_nan[2, 1])
    X_fill, _ = extract_table(missing_tbl; features=[:x1, :x2], missing_policy=(:fill => -1.0))
    @test X_fill[2, 1] == -1.0
end
