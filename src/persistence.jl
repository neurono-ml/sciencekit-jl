function _validate_path(path::AbstractString)::String
    s = String(path)
    if isempty(s)
        throw(ScienceKitError(STATUS_INVALID_ARGUMENT, "file path must not be empty"))
    end
    if occursin('\0', s)
        throw(ScienceKitError(STATUS_INVALID_ARGUMENT, "file path must not contain interior NUL bytes"))
    end
    return s
end

"""
    save(model::Model, path::AbstractString) -> Nothing

Persist a fitted `Model` to `path` via `sk_ffi_model_save`.
Currently supports `Float64` `LinearRegression` models per the C ABI contract;
unsupported configurations raise `ScienceKitError(STATUS_UNSUPPORTED, ...)`.
"""
function save(model::Model, path::AbstractString)::Nothing
    clean_path = _validate_path(path)
    _with_model_handle(model) do m_ptr
        code = ccall(_FFI.model_save, Int32, (Ptr{Cvoid}, Cstring), m_ptr, clean_path)
        _check_status(code)
    end
    return nothing
end

"""
    load(path::AbstractString) -> Model

Load a persisted `Model` from `path` via `sk_ffi_model_load`.
"""
function load(path::AbstractString)::Model
    _require_ffi_loaded()
    clean_path = _validate_path(path)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    code = ccall(_FFI.model_load, Int32, (Cstring, Ref{Ptr{Cvoid}}), clean_path, out_ptr)
    _check_status(code)
    if out_ptr[] == C_NULL
        throw(ScienceKitError(STATUS_INTERNAL, "sk_ffi_model_load succeeded with a null model handle"))
    end
    return Model(out_ptr[], nothing)
end
