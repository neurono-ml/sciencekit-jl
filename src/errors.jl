"""
Stable status codes returned by the `sciencekit_ffi` C ABI (`SKFfiStatus`).
"""
@enum Status::Int32 begin
    STATUS_OK = 0
    STATUS_INVALID_HANDLE = 1
    STATUS_SHAPE = 2
    STATUS_INVALID_ARGUMENT = 3
    STATUS_UNSUPPORTED = 4
    STATUS_NOT_FITTED = 5
    STATUS_IO = 6
    STATUS_INTERNAL = 7
end

"""
    ScienceKitError(status::Status, message::AbstractString)

Structured exception raised whenever an input validation check or a native
`sciencekit_ffi` entry point fails. Carries the typed `status` (`Status`),
its numeric ABI `code` (`Int32`), and a copied human-readable `message`.
"""
struct ScienceKitError <: Exception
    status::Status
    code::Int32
    message::String

    function ScienceKitError(status::Status, message::AbstractString)
        new(status, Int32(status), String(message))
    end

    function ScienceKitError(code::Integer, message::AbstractString)
        status = _try_status_from_code(Int32(code))
        new(status, Int32(code), String(message))
    end
end

function _try_status_from_code(code::Int32)::Status
    code == Int32(0) && return STATUS_OK
    code == Int32(1) && return STATUS_INVALID_HANDLE
    code == Int32(2) && return STATUS_SHAPE
    code == Int32(3) && return STATUS_INVALID_ARGUMENT
    code == Int32(4) && return STATUS_UNSUPPORTED
    code == Int32(5) && return STATUS_NOT_FITTED
    code == Int32(6) && return STATUS_IO
    return STATUS_INTERNAL
end

function _fallback_status_name(status::Status)::String
    status == STATUS_OK && return "ok"
    status == STATUS_INVALID_HANDLE && return "invalid-handle"
    status == STATUS_SHAPE && return "shape"
    status == STATUS_INVALID_ARGUMENT && return "invalid-argument"
    status == STATUS_UNSUPPORTED && return "unsupported"
    status == STATUS_NOT_FITTED && return "not-fitted"
    status == STATUS_IO && return "io"
    return "internal"
end

function Base.showerror(io::IO, err::ScienceKitError)
    label = _is_ffi_loaded() ? status_name(err.status) : _fallback_status_name(err.status)
    if isempty(err.message)
        print(io, "ScienceKitError(", label, " [code=", err.code, "])")
    else
        print(io, "ScienceKitError(", label, " [code=", err.code, "]): ", err.message)
    end
end
