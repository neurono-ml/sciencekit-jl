"""
    ConversionReport

Describes how a Julia matrix or vector was prepared for the row-major
`sciencekit_ffi` C ABI boundary:
- `path`: `:borrowed_contiguous` (zero-copy borrow) or `:transposed_copy` (column-major to row-major copy)
- `shape`: logical `(rows, columns)` or `(length,)`
- `dtype`: `Float64` or `Float32`
- `copied`: `true` when an intermediate buffer had to be allocated
"""
struct ConversionReport
    path::Symbol
    shape::Tuple{Vararg{Int}}
    dtype::DataType
    copied::Bool
end

struct _PreparedBuffer{T<:Union{Float32, Float64}}
    backing::Array{T}
    shape::Vector{Int64}
    dtype_code::DataTypeCode
    report::ConversionReport
end

_dtype_code_for_eltype(::Type{Float64}) = DTYPE_F64
_dtype_code_for_eltype(::Type{Float32}) = DTYPE_F32

function _validate_float_eltype(::Type{T}, role::AbstractString) where {T}
    if T === Float64 || T === Float32
        return nothing
    end
    throw(ScienceKitError(
        STATUS_INVALID_ARGUMENT,
        "$(role) has unsupported element type $(T); convert explicitly with Float64.(x) or Float32.(x)"
    ))
end

function _prepare_matrix(X::AbstractMatrix{T})::_PreparedBuffer{T} where {T<:Union{Float32, Float64}}
    rows, cols = size(X)
    if rows <= 0 || cols <= 0
        throw(ScienceKitError(
            STATUS_SHAPE,
            "feature matrix must have strictly positive dimensions, got ($(rows), $(cols))"
        ))
    end

    shape_vec = Int64[Int64(rows), Int64(cols)]
    dtype_code = _dtype_code_for_eltype(T)

    # Zero-copy case 1: contiguous Matrix{T} with a single row or a single column
    if X isa Matrix{T} && (rows == 1 || cols == 1)
        report = ConversionReport(:borrowed_contiguous, (rows, cols), T, false)
        return _PreparedBuffer{T}(X, shape_vec, dtype_code, report)
    end

    # Zero-copy case 2: transpose/adjoint wrapper around a dense Matrix{T}
    # For real floats (Float32/Float64), adjoint == transpose, and parent(X) has
    # column-major size (cols, rows) whose flat buffer is exact C row-major for (rows, cols).
    if hasmethod(parent, Tuple{typeof(X)})
        p = parent(X)
        if p isa Matrix{T} && size(p) == (cols, rows) && strides(X) == (cols, 1)
            report = ConversionReport(:borrowed_contiguous, (rows, cols), T, false)
            return _PreparedBuffer{T}(p, shape_vec, dtype_code, report)
        end
    end

    # General column-major or strided AbstractMatrix{T}: materialize row-major via permutedims
    row_major_backing = Matrix{T}(permutedims(X))
    report = ConversionReport(:transposed_copy, (rows, cols), T, true)
    return _PreparedBuffer{T}(row_major_backing, shape_vec, dtype_code, report)
end

function _prepare_matrix(X::AbstractMatrix{T}) where {T}
    _validate_float_eltype(T, "feature matrix")
end

function _prepare_matrix(X)
    throw(ScienceKitError(
        STATUS_SHAPE,
        "features must be a 2-dimensional AbstractMatrix{Float64} or AbstractMatrix{Float32}, got $(typeof(X))"
    ))
end

function _prepare_vector(y::AbstractVector{T})::_PreparedBuffer{T} where {T<:Union{Float32, Float64}}
    n = length(y)
    if n <= 0
        throw(ScienceKitError(
            STATUS_SHAPE,
            "target vector must be non-empty, got length $(n)"
        ))
    end

    shape_vec = Int64[Int64(n)]
    dtype_code = _dtype_code_for_eltype(T)

    if y isa Vector{T}
        report = ConversionReport(:borrowed_contiguous, (n,), T, false)
        return _PreparedBuffer{T}(y, shape_vec, dtype_code, report)
    end

    contiguous = collect(T, y)
    report = ConversionReport(:transposed_copy, (n,), T, true)
    return _PreparedBuffer{T}(contiguous, shape_vec, dtype_code, report)
end

function _prepare_vector(y::AbstractVector{T}) where {T}
    _validate_float_eltype(T, "target vector")
end

function _prepare_vector(y)
    throw(ScienceKitError(
        STATUS_SHAPE,
        "targets must be a 1-dimensional AbstractVector{Float64} or AbstractVector{Float32}, got $(typeof(y))"
    ))
end

"""
    inspect_matrix_conversion(X) -> ConversionReport

Validate `X` as a feature matrix and return a `ConversionReport` describing
whether `ScienceKit.jl` borrows `X` zero-copy or transposes it into a row-major buffer.
"""
inspect_matrix_conversion(X) = _prepare_matrix(X).report

"""
    inspect_vector_conversion(y) -> ConversionReport

Validate `y` as a target vector and return its `ConversionReport`.
"""
inspect_vector_conversion(y) = _prepare_vector(y).report

function _with_descriptor(f::F, prepared::_PreparedBuffer{T}) where {F, T}
    backing = prepared.backing
    shape_vec = prepared.shape
    GC.@preserve backing shape_vec begin
        desc = FfiArrayDescriptor(
            Int32(prepared.dtype_code),
            Int32(length(shape_vec)),
            pointer(shape_vec),
            C_NULL,
            Ptr{Cvoid}(pointer(backing)),
            Int64(length(backing)),
        )
        desc_ref = Ref(desc)
        GC.@preserve desc_ref begin
            return f(desc_ref)
        end
    end
end

function _copy_output_vector(::Type{T}, data_ptr::Ptr{Cvoid}, len::Int64)::Vector{T} where {T}
    out = Vector{T}(undef, Int(len))
    if len > 0
        unsafe_copyto!(pointer(out), Ptr{T}(data_ptr), Int(len))
    end
    return out
end

function _copy_output_matrix(::Type{T}, data_ptr::Ptr{Cvoid}, rows::Int64, cols::Int64)::Matrix{T} where {T}
    total = Int(rows * cols)
    # Native buffer is C row-major (rows, cols), which corresponds to a Julia
    # column-major buffer of shape (cols, rows) before `permutedims`.
    row_major_stage = Matrix{T}(undef, Int(cols), Int(rows))
    if total > 0
        unsafe_copyto!(pointer(row_major_stage), Ptr{T}(data_ptr), total)
    end
    return permutedims(row_major_stage)
end

function _consume_output_array!(desc_ref::Ref{FfiArrayDescriptor}; expect_ndim::Int)
    try
        desc = desc_ref[]
        if desc.data == C_NULL || desc.shape == C_NULL
            throw(ScienceKitError(STATUS_INTERNAL, "native call returned a null output buffer"))
        end
        if Int(desc.ndim) != expect_ndim
            throw(ScienceKitError(
                STATUS_INTERNAL,
                "expected $(expect_ndim)-D output array from native ABI, got $(desc.ndim)-D"
            ))
        end

        T = if desc.dtype == Int32(DTYPE_F64)
            Float64
        elseif desc.dtype == Int32(DTYPE_F32)
            Float32
        else
            throw(ScienceKitError(
                STATUS_UNSUPPORTED,
                "unsupported output dtype code $(desc.dtype)"
            ))
        end

        if expect_ndim == 1
            n = unsafe_load(desc.shape, 1)
            return _copy_output_vector(T, desc.data, n)
        else
            rows = unsafe_load(desc.shape, 1)
            cols = unsafe_load(desc.shape, 2)
            return _copy_output_matrix(T, desc.data, rows, cols)
        end
    finally
        if _is_ffi_loaded()
            ccall(_FFI.array_free, Int32, (Ref{FfiArrayDescriptor},), desc_ref)
        end
    end
end
