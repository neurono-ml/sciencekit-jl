const EXPECTED_ABI_MAJOR = Int32(1)

const FEATURE_LINEAR_REGRESSION = UInt64(1) << 0
const FEATURE_STANDARD_SCALER = UInt64(1) << 1
const FEATURE_MODEL_PERSISTENCE = UInt64(1) << 2
const FEATURE_F32 = UInt64(1) << 3
const FEATURE_F64 = UInt64(1) << 4

@enum DataTypeCode::Int32 begin
    DTYPE_F32 = 0
    DTYPE_F64 = 1
    DTYPE_I64 = 2
    DTYPE_U8 = 3
end

@enum EstimatorFamily::Int32 begin
    ESTIMATOR_LINEAR_REGRESSION = 0
    ESTIMATOR_STANDARD_SCALER = 1
end

@enum BackendKind::Int32 begin
    BACKEND_FAER = 0
    BACKEND_MATRIX_MULTIPLY = 1
    BACKEND_NDARRAY_LINALG = 2
end

"""
Mirror of C `SKFfiArray` (`crates/sciencekit_ffi/include/sciencekit.h`).
"""
struct FfiArrayDescriptor
    dtype::Int32
    ndim::Int32
    shape::Ptr{Int64}
    strides::Ptr{Int64}
    data::Ptr{Cvoid}
    len::Int64
end

_empty_descriptor() = FfiArrayDescriptor(Int32(0), Int32(0), C_NULL, C_NULL, C_NULL, Int64(0))

Base.@kwdef mutable struct _FfiSymbols
    lib::Ptr{Cvoid} = C_NULL
    library_path::String = ""
    version::Ptr{Cvoid} = C_NULL
    features::Ptr{Cvoid} = C_NULL
    status_name::Ptr{Cvoid} = C_NULL
    last_error::Ptr{Cvoid} = C_NULL
    builder_create::Ptr{Cvoid} = C_NULL
    builder_set_include_intercept::Ptr{Cvoid} = C_NULL
    builder_set_with_mean::Ptr{Cvoid} = C_NULL
    builder_set_with_std::Ptr{Cvoid} = C_NULL
    builder_set_backend::Ptr{Cvoid} = C_NULL
    builder_free::Ptr{Cvoid} = C_NULL
    fit::Ptr{Cvoid} = C_NULL
    model_free::Ptr{Cvoid} = C_NULL
    predict::Ptr{Cvoid} = C_NULL
    transform::Ptr{Cvoid} = C_NULL
    score::Ptr{Cvoid} = C_NULL
    model_save::Ptr{Cvoid} = C_NULL
    model_load::Ptr{Cvoid} = C_NULL
    array_free::Ptr{Cvoid} = C_NULL
end

const _FFI = _FfiSymbols()

_is_ffi_loaded() = _FFI.lib != C_NULL

function _platform_library_name()::String
    if Sys.isapple()
        return "libsciencekit_ffi.dylib"
    elseif Sys.iswindows()
        return "sciencekit_ffi.dll"
    else
        return "libsciencekit_ffi.so"
    end
end

function _discover_library_candidates()::Vector{String}
    candidates = String[]
    libname = _platform_library_name()

    explicit_file = strip(get(ENV, "SCIENCEKIT_FFI_LIB", ""))
    !isempty(explicit_file) && push!(candidates, explicit_file)

    explicit_dir = strip(get(ENV, "SCIENCEKIT_LIB_DIR", ""))
    !isempty(explicit_dir) && push!(candidates, joinpath(explicit_dir, libname))

    bundled = normpath(joinpath(@__DIR__, "..", "lib", libname))
    push!(candidates, bundled)

    push!(candidates, libname)
    push!(candidates, "libsciencekit_ffi")
    return candidates
end

function _init_ffi!()
    candidates = _discover_library_candidates()
    handle = C_NULL
    chosen_path = ""

    for candidate in candidates
        if isabspath(candidate) && !isfile(candidate)
            continue
        end
        opened = Libdl.dlopen(candidate, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL; throw_error=false)
        if opened !== nothing && opened != C_NULL
            handle = opened
            chosen_path = candidate
            break
        end
    end

    if handle == C_NULL
        throw(ScienceKitError(
            STATUS_IO,
            "could not load the native sciencekit_ffi shared library; run " *
            "`scripts/build-native-library.sh /path/to/sciencekit` or set " *
            "`SCIENCEKIT_FFI_LIB` to the compiled shared library path (searched: " *
            join(candidates, ", ") * ")"
        ))
    end

    _FFI.lib = handle
    _FFI.library_path = chosen_path
    _FFI.version = Libdl.dlsym(handle, :sk_ffi_version)
    _FFI.features = Libdl.dlsym(handle, :sk_ffi_features)
    _FFI.status_name = Libdl.dlsym(handle, :sk_ffi_status_name)
    _FFI.last_error = Libdl.dlsym(handle, :sk_ffi_last_error)
    _FFI.builder_create = Libdl.dlsym(handle, :sk_ffi_builder_create)
    _FFI.builder_set_include_intercept = Libdl.dlsym(handle, :sk_ffi_builder_set_include_intercept)
    _FFI.builder_set_with_mean = Libdl.dlsym(handle, :sk_ffi_builder_set_with_mean)
    _FFI.builder_set_with_std = Libdl.dlsym(handle, :sk_ffi_builder_set_with_std)
    _FFI.builder_set_backend = Libdl.dlsym(handle, :sk_ffi_builder_set_backend)
    _FFI.builder_free = Libdl.dlsym(handle, :sk_ffi_builder_free)
    _FFI.fit = Libdl.dlsym(handle, :sk_ffi_fit)
    _FFI.model_free = Libdl.dlsym(handle, :sk_ffi_model_free)
    _FFI.predict = Libdl.dlsym(handle, :sk_ffi_predict)
    _FFI.transform = Libdl.dlsym(handle, :sk_ffi_transform)
    _FFI.score = Libdl.dlsym(handle, :sk_ffi_score)
    _FFI.model_save = Libdl.dlsym(handle, :sk_ffi_model_save)
    _FFI.model_load = Libdl.dlsym(handle, :sk_ffi_model_load)
    _FFI.array_free = Libdl.dlsym(handle, :sk_ffi_array_free)

    ver = abi_version()
    if Int32(ver.major) != EXPECTED_ABI_MAJOR
        throw(ScienceKitError(
            STATUS_UNSUPPORTED,
            "incompatible sciencekit_ffi ABI major version $(ver.major) (expected $(EXPECTED_ABI_MAJOR))"
        ))
    end
    return nothing
end

function _require_ffi_loaded()
    if !_is_ffi_loaded()
        throw(ScienceKitError(STATUS_INTERNAL, "ScienceKit FFI symbols are not initialized"))
    end
    return nothing
end

function _last_error_message()::String
    !_is_ffi_loaded() && return ""
    ptr = ccall(_FFI.last_error, Ptr{Cchar}, ())
    ptr == C_NULL && return ""
    return unsafe_string(ptr)
end

function _check_status(raw_code::Int32)
    raw_code == Int32(STATUS_OK) && return nothing
    msg = _last_error_message()
    throw(ScienceKitError(raw_code, msg))
end

"""
    library_path() -> String

Return the resolved path or loader name of the loaded `sciencekit_ffi` library.
"""
function library_path()::String
    _require_ffi_loaded()
    return _FFI.library_path
end

"""
    abi_version() -> VersionNumber

Query `sk_ffi_version` and return the native C ABI semantic version.
"""
function abi_version()::VersionNumber
    _require_ffi_loaded()
    major = Ref{Int32}(-1)
    minor = Ref{Int32}(-1)
    patch = Ref{Int32}(-1)
    code = ccall(_FFI.version, Int32, (Ref{Int32}, Ref{Int32}, Ref{Int32}), major, minor, patch)
    _check_status(code)
    return VersionNumber(Int(major[]), Int(minor[]), Int(patch[]))
end

"""
    features() -> UInt64

Return the capability bitmask advertised by `sk_ffi_features`.
"""
function features()::UInt64
    _require_ffi_loaded()
    return ccall(_FFI.features, UInt64, ())
end

"""
    has_feature(flag::UInt64) -> Bool

Return `true` when every bit in `flag` is advertised by `features()`.
"""
has_feature(flag::UInt64)::Bool = (features() & flag) == flag

"""
    status_name(status::Union{Status, Integer}) -> String

Return the canonical lowercase name for an ABI status code (`"ok"`, `"shape"`, ...).
"""
function status_name(status::Union{Status, Integer})::String
    _require_ffi_loaded()
    ptr = ccall(_FFI.status_name, Ptr{Cchar}, (Int32,), Int32(status))
    ptr == C_NULL && return "unknown"
    return unsafe_string(ptr)
end
