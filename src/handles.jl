const _LIVE_HANDLES = Threads.Atomic{Int}(0)

"""
    live_handle_count() -> Int

Return the number of native `SKFfiBuilder` and `SKFfiModel` handles currently
allocated by this Julia process and not yet released via `close!` or GC finalization.
"""
live_handle_count()::Int = _LIVE_HANDLES[]

function _resolve_dtype_code(dtype::DataTypeCode)::DataTypeCode
    return dtype
end

function _resolve_dtype_code(dtype::Type)::DataTypeCode
    dtype === Float64 && return DTYPE_F64
    dtype === Float32 && return DTYPE_F32
    throw(ScienceKitError(
        STATUS_INVALID_ARGUMENT,
        "unsupported estimator dtype $(dtype); expected Float64 or Float32"
    ))
end

function _resolve_backend_code(backend::BackendKind)::Int32
    return Int32(backend)
end

function _resolve_backend_code(backend::Integer)::Int32
    return Int32(backend)
end

function _resolve_backend_code(backend::Symbol)::Int32
    backend === :faer && return Int32(BACKEND_FAER)
    (backend === :matrix_multiply || backend === :matrixmultiply) && return Int32(BACKEND_MATRIX_MULTIPLY)
    (backend === :ndarray_linalg || backend === :ndarray) && return Int32(BACKEND_NDARRAY_LINALG)
    throw(ScienceKitError(
        STATUS_INVALID_ARGUMENT,
        "unknown backend :$(backend); expected :faer, :matrix_multiply, or :ndarray_linalg"
    ))
end

"""
    Builder(family::EstimatorFamily, dtype=Float64)

Mutable handle wrapping a native `SKFfiBuilder *` pointer. Supports explicit,
idempotent `close!(builder)` as well as automatic release via a GC finalizer.
"""
mutable struct Builder
    const spin::Threads.SpinLock
    const family::EstimatorFamily
    const dtype::DataTypeCode
    handle::Ptr{Cvoid}
    in_flight::Int
    closed::Bool

    function Builder(family::EstimatorFamily, dtype::Union{DataTypeCode, Type}=Float64)
        _require_ffi_loaded()
        dtype_code = _resolve_dtype_code(dtype)
        ptr = ccall(_FFI.builder_create, Ptr{Cvoid}, (Int32, Int32), Int32(family), Int32(dtype_code))
        if ptr == C_NULL
            throw(ScienceKitError(STATUS_INVALID_ARGUMENT, _last_error_message()))
        end
        Threads.atomic_add!(_LIVE_HANDLES, 1)
        obj = new(Threads.SpinLock(), family, dtype_code, ptr, 0, false)
        finalizer(_finalize_builder!, obj)
        return obj
    end
end

"""
    Model

Opaque handle wrapping a fitted or loaded native `SKFfiModel *` pointer.
Fitted models are immutable on the Rust side and safe to share across Julia
threads for concurrent `predict`, `transform`, `score`, and `save` calls.
"""
mutable struct Model
    const spin::Threads.SpinLock
    const dtype::Union{DataTypeCode, Nothing}
    handle::Ptr{Cvoid}
    in_flight::Int
    closed::Bool

    function Model(ptr::Ptr{Cvoid}, dtype::Union{DataTypeCode, Nothing}=nothing)
        if ptr == C_NULL
            throw(ScienceKitError(STATUS_INVALID_HANDLE, "cannot wrap a null model handle"))
        end
        Threads.atomic_add!(_LIVE_HANDLES, 1)
        obj = new(Threads.SpinLock(), dtype, ptr, 0, false)
        finalizer(_finalize_model!, obj)
        return obj
    end
end

function _free_builder_ptr!(ptr::Ptr{Cvoid}; check::Bool)
    ptr == C_NULL && return nothing
    code = ccall(_FFI.builder_free, Int32, (Ptr{Cvoid},), ptr)
    Threads.atomic_sub!(_LIVE_HANDLES, 1)
    check && _check_status(code)
    return nothing
end

function _free_model_ptr!(ptr::Ptr{Cvoid}; check::Bool)
    ptr == C_NULL && return nothing
    code = ccall(_FFI.model_free, Int32, (Ptr{Cvoid},), ptr)
    Threads.atomic_sub!(_LIVE_HANDLES, 1)
    check && _check_status(code)
    return nothing
end

function _finalize_builder!(builder::Builder)
    if _is_ffi_loaded()
        _close_builder!(builder; check=false)
    end
    return nothing
end

function _finalize_model!(model::Model)
    if _is_ffi_loaded()
        _close_model!(model; check=false)
    end
    return nothing
end

function _close_builder!(builder::Builder; check::Bool)
    to_free = C_NULL
    lock(builder.spin)
    try
        if builder.closed
            return nothing
        end
        builder.closed = true
        if builder.in_flight == 0 && builder.handle != C_NULL
            to_free = builder.handle
            builder.handle = C_NULL
        end
    finally
        unlock(builder.spin)
    end
    _free_builder_ptr!(to_free; check=check)
    return nothing
end

function _close_model!(model::Model; check::Bool)
    to_free = C_NULL
    lock(model.spin)
    try
        if model.closed
            return nothing
        end
        model.closed = true
        if model.in_flight == 0 && model.handle != C_NULL
            to_free = model.handle
            model.handle = C_NULL
        end
    finally
        unlock(model.spin)
    end
    _free_model_ptr!(to_free; check=check)
    return nothing
end

"""
    close!(handle::Union{Builder, Model}) -> Nothing

Release the underlying native C handle immediately. Safe to call any number of
times (idempotent): subsequent `close!` calls are no-ops, while any subsequent
operation on `handle` raises `ScienceKitError(STATUS_INVALID_HANDLE, ...)`.
"""
close!(builder::Builder) = _close_builder!(builder; check=true)
close!(model::Model) = _close_model!(model; check=true)

Base.close(builder::Builder) = close!(builder)
Base.close(model::Model) = close!(model)

"""
    is_closed(handle::Union{Builder, Model}) -> Bool

Return `true` if `close!` has already been called on `handle`.
"""
function is_closed(handle::Union{Builder, Model})::Bool
    lock(handle.spin)
    try
        return handle.closed
    finally
        unlock(handle.spin)
    end
end

function _with_builder_handle(f::F, builder::Builder) where {F}
    ptr = C_NULL
    lock(builder.spin)
    try
        if builder.closed || builder.handle == C_NULL
            throw(ScienceKitError(STATUS_INVALID_HANDLE, "builder handle is closed"))
        end
        builder.in_flight += 1
        ptr = builder.handle
    finally
        unlock(builder.spin)
    end

    deferred_free = C_NULL
    try
        return f(ptr)
    finally
        lock(builder.spin)
        try
            builder.in_flight -= 1
            if builder.in_flight == 0 && builder.closed && builder.handle != C_NULL
                deferred_free = builder.handle
                builder.handle = C_NULL
            end
        finally
            unlock(builder.spin)
        end
        _free_builder_ptr!(deferred_free; check=false)
    end
end

function _with_model_handle(f::F, model::Model) where {F}
    ptr = C_NULL
    lock(model.spin)
    try
        if model.closed || model.handle == C_NULL
            throw(ScienceKitError(STATUS_INVALID_HANDLE, "model handle is closed"))
        end
        model.in_flight += 1
        ptr = model.handle
    finally
        unlock(model.spin)
    end

    deferred_free = C_NULL
    try
        return f(ptr)
    finally
        lock(model.spin)
        try
            model.in_flight -= 1
            if model.in_flight == 0 && model.closed && model.handle != C_NULL
                deferred_free = model.handle
                model.handle = C_NULL
            end
        finally
            unlock(model.spin)
        end
        _free_model_ptr!(deferred_free; check=false)
    end
end

"""
    set_include_intercept!(builder::Builder, enabled::Bool) -> Builder

Configure whether `LinearRegression` fits an intercept term.
"""
function set_include_intercept!(builder::Builder, enabled::Bool)::Builder
    _with_builder_handle(builder) do ptr
        code = ccall(_FFI.builder_set_include_intercept, Int32, (Ptr{Cvoid}, Bool), ptr, enabled)
        _check_status(code)
    end
    return builder
end

"""
    set_with_mean!(builder::Builder, enabled::Bool) -> Builder

Configure whether `StandardScaler` centers features by subtracting column means.
"""
function set_with_mean!(builder::Builder, enabled::Bool)::Builder
    _with_builder_handle(builder) do ptr
        code = ccall(_FFI.builder_set_with_mean, Int32, (Ptr{Cvoid}, Bool), ptr, enabled)
        _check_status(code)
    end
    return builder
end

"""
    set_with_std!(builder::Builder, enabled::Bool) -> Builder

Configure whether `StandardScaler` scales features to unit variance.
"""
function set_with_std!(builder::Builder, enabled::Bool)::Builder
    _with_builder_handle(builder) do ptr
        code = ccall(_FFI.builder_set_with_std, Int32, (Ptr{Cvoid}, Bool), ptr, enabled)
        _check_status(code)
    end
    return builder
end

"""
    set_backend!(builder::Builder, backend) -> Builder

Select the math backend (`:faer`, `:matrix_multiply`, `:ndarray_linalg`, or a `BackendKind`).
"""
function set_backend!(builder::Builder, backend::Union{BackendKind, Symbol, Integer})::Builder
    backend_code = _resolve_backend_code(backend)
    _with_builder_handle(builder) do ptr
        code = ccall(_FFI.builder_set_backend, Int32, (Ptr{Cvoid}, Int32), ptr, backend_code)
        _check_status(code)
    end
    return builder
end

"""
    linear_regression_builder(; dtype=Float64, include_intercept=true, backend=:faer) -> Builder

Create and configure an ordinary-least-squares linear regression `Builder`.
"""
function linear_regression_builder(;
    dtype::Union{DataTypeCode, Type}=Float64,
    include_intercept::Bool=true,
    backend::Union{BackendKind, Symbol, Integer}=BACKEND_FAER,
)::Builder
    builder = Builder(ESTIMATOR_LINEAR_REGRESSION, dtype)
    try
        set_include_intercept!(builder, include_intercept)
        set_backend!(builder, backend)
    catch
        close!(builder)
        rethrow()
    end
    return builder
end

"""
    standard_scaler_builder(; dtype=Float64, with_mean=true, with_std=true, backend=:faer) -> Builder

Create and configure a `StandardScaler` `Builder`.
"""
function standard_scaler_builder(;
    dtype::Union{DataTypeCode, Type}=Float64,
    with_mean::Bool=true,
    with_std::Bool=true,
    backend::Union{BackendKind, Symbol, Integer}=BACKEND_FAER,
)::Builder
    builder = Builder(ESTIMATOR_STANDARD_SCALER, dtype)
    try
        set_with_mean!(builder, with_mean)
        set_with_std!(builder, with_std)
        set_backend!(builder, backend)
    catch
        close!(builder)
        rethrow()
    end
    return builder
end
