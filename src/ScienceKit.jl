module ScienceKit

using Libdl

export Status,
    STATUS_OK,
    STATUS_INVALID_HANDLE,
    STATUS_SHAPE,
    STATUS_INVALID_ARGUMENT,
    STATUS_UNSUPPORTED,
    STATUS_NOT_FITTED,
    STATUS_IO,
    STATUS_INTERNAL,
    ScienceKitError,
    DataTypeCode,
    DTYPE_F32,
    DTYPE_F64,
    DTYPE_I64,
    DTYPE_U8,
    EstimatorFamily,
    ESTIMATOR_LINEAR_REGRESSION,
    ESTIMATOR_STANDARD_SCALER,
    BackendKind,
    BACKEND_FAER,
    BACKEND_MATRIX_MULTIPLY,
    BACKEND_NDARRAY_LINALG,
    FEATURE_LINEAR_REGRESSION,
    FEATURE_STANDARD_SCALER,
    FEATURE_MODEL_PERSISTENCE,
    FEATURE_F32,
    FEATURE_F64,
    abi_version,
    features,
    has_feature,
    status_name,
    library_path

include("errors.jl")
include("ffi.jl")

function __init__()
    _init_ffi!()
end

end # module ScienceKit
