function _classify_fit_error(msg::AbstractString)::Status
    lower = lowercase(msg)
    occursin("invalid handle", lower) && return STATUS_INVALID_HANDLE
    (occursin("shape", lower) || occursin("dimension", lower) || occursin("length", lower)) && return STATUS_SHAPE
    (occursin("not supported", lower) || occursin("unsupported", lower)) && return STATUS_UNSUPPORTED
    occursin("not been fitted", lower) && return STATUS_NOT_FITTED
    occursin("i/o error", lower) && return STATUS_IO
    (occursin("panic:", lower) || occursin("internal error", lower)) && return STATUS_INTERNAL
    return STATUS_INVALID_ARGUMENT
end

function _expected_eltype(code::DataTypeCode)::Type
    code == DTYPE_F64 && return Float64
    code == DTYPE_F32 && return Float32
    throw(ScienceKitError(STATUS_UNSUPPORTED, "unsupported builder dtype $(code)"))
end

"""
    fit(builder::Builder, features, targets=nothing) -> Model

Fit the configured estimator on `features` (`AbstractMatrix{Float64}` or
`AbstractMatrix{Float32}`) and optional `targets` (`AbstractVector{Float64}` or
`AbstractVector{Float32}`). Returns a fitted `Model` handle.
"""
function fit(builder::Builder, features, targets=nothing)::Model
    prep_X = _prepare_matrix(features)
    if prep_X.dtype_code != builder.dtype
        expected_T = _expected_eltype(builder.dtype)
        throw(ScienceKitError(
            STATUS_INVALID_ARGUMENT,
            "feature matrix element type $(prep_X.report.dtype) does not match builder dtype $(expected_T)"
        ))
    end

    if builder.family == ESTIMATOR_LINEAR_REGRESSION && targets === nothing
        throw(ScienceKitError(
            STATUS_INVALID_ARGUMENT,
            "linear regression requires a target vector `y`"
        ))
    end

    if targets === nothing
        model_ptr = _with_builder_handle(builder) do b_ptr
            _with_descriptor(prep_X) do x_ref
                ccall(
                    _FFI.fit,
                    Ptr{Cvoid},
                    (Ptr{Cvoid}, Ref{FfiArrayDescriptor}, Ptr{Cvoid}),
                    b_ptr,
                    x_ref,
                    C_NULL,
                )
            end
        end
        if model_ptr == C_NULL
            msg = _last_error_message()
            throw(ScienceKitError(_classify_fit_error(msg), msg))
        end
        return Model(model_ptr, builder.dtype)
    else
        prep_y = _prepare_vector(targets)
        if prep_y.dtype_code != builder.dtype
            expected_T = _expected_eltype(builder.dtype)
            throw(ScienceKitError(
                STATUS_INVALID_ARGUMENT,
                "target vector element type $(prep_y.report.dtype) does not match builder dtype $(expected_T)"
            ))
        end
        if prep_X.shape[1] != prep_y.shape[1]
            throw(ScienceKitError(
                STATUS_SHAPE,
                "feature row count ($(prep_X.shape[1])) does not match target length ($(prep_y.shape[1]))"
            ))
        end

        model_ptr = _with_builder_handle(builder) do b_ptr
            _with_descriptor(prep_X) do x_ref
                _with_descriptor(prep_y) do y_ref
                    ccall(
                        _FFI.fit,
                        Ptr{Cvoid},
                        (Ptr{Cvoid}, Ref{FfiArrayDescriptor}, Ref{FfiArrayDescriptor}),
                        b_ptr,
                        x_ref,
                        y_ref,
                    )
                end
            end
        end
        if model_ptr == C_NULL
            msg = _last_error_message()
            throw(ScienceKitError(_classify_fit_error(msg), msg))
        end
        return Model(model_ptr, builder.dtype)
    end
end

"""
    fit_table(builder::Builder, table; features=nothing, target=nothing, missing_policy=:reject) -> Model

Extract numeric columns from `table` matching `builder`'s scalar precision and
fit a `Model`.
"""
function fit_table(
    builder::Builder,
    table;
    features=nothing,
    target::Union{Symbol, AbstractString, Nothing}=nothing,
    missing_policy=:reject,
)::Model
    T = _expected_eltype(builder.dtype)
    X, y = extract_table(table; features=features, target=target, dtype=T, missing_policy=missing_policy)
    return fit(builder, X, y)
end

"""
    predict(model::Model, features) -> Vector

Predict target values for `features`, returning `Vector{Float64}` or `Vector{Float32}`
matching the fitted model's precision.
"""
function predict(model::Model, features)
    prep_X = _prepare_matrix(features)
    return _with_model_handle(model) do m_ptr
        out_ref = Ref(_empty_descriptor())
        code = _with_descriptor(prep_X) do x_ref
            ccall(
                _FFI.predict,
                Int32,
                (Ptr{Cvoid}, Ref{FfiArrayDescriptor}, Ref{FfiArrayDescriptor}),
                m_ptr,
                x_ref,
                out_ref,
            )
        end
        _check_status(code)
        return _consume_output_array!(out_ref; expect_ndim=1)
    end
end

"""
    transform(model::Model, features) -> Matrix

Transform `features` with a fitted transformer (such as `StandardScaler`),
returning a column-major Julia `Matrix{Float64}` or `Matrix{Float32}` of shape `(rows, cols)`.
"""
function transform(model::Model, features)
    prep_X = _prepare_matrix(features)
    return _with_model_handle(model) do m_ptr
        out_ref = Ref(_empty_descriptor())
        code = _with_descriptor(prep_X) do x_ref
            ccall(
                _FFI.transform,
                Int32,
                (Ptr{Cvoid}, Ref{FfiArrayDescriptor}, Ref{FfiArrayDescriptor}),
                m_ptr,
                x_ref,
                out_ref,
            )
        end
        _check_status(code)
        return _consume_output_array!(out_ref; expect_ndim=2)
    end
end

"""
    score(model::Model, features, targets) -> Float64

Evaluate `model` on `features` and `targets` (returning `R²` for regression models).
"""
function score(model::Model, features, targets)::Float64
    prep_X = _prepare_matrix(features)
    prep_y = _prepare_vector(targets)
    if prep_X.shape[1] != prep_y.shape[1]
        throw(ScienceKitError(
            STATUS_SHAPE,
            "feature row count ($(prep_X.shape[1])) does not match target length ($(prep_y.shape[1]))"
        ))
    end

    return _with_model_handle(model) do m_ptr
        score_ref = Ref{Float64}(0.0)
        code = _with_descriptor(prep_X) do x_ref
            _with_descriptor(prep_y) do y_ref
                ccall(
                    _FFI.score,
                    Int32,
                    (Ptr{Cvoid}, Ref{FfiArrayDescriptor}, Ref{FfiArrayDescriptor}, Ref{Float64}),
                    m_ptr,
                    x_ref,
                    y_ref,
                    score_ref,
                )
            end
        end
        _check_status(code)
        return score_ref[]
    end
end
