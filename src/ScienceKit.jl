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
    library_path,
    Builder,
    Model,
    linear_regression_builder,
    standard_scaler_builder,
    set_include_intercept!,
    set_with_mean!,
    set_with_std!,
    set_backend!,
    close!,
    is_closed,
    live_handle_count,
    ConversionReport,
    inspect_matrix_conversion,
    inspect_vector_conversion,
    extract_table,
    fit,
    fit_table,
    predict,
    transform,
    score,
    save,
    load

include("errors.jl")
include("ffi.jl")
include("handles.jl")
include("arrays.jl")
include("tables.jl")
include("operations.jl")
include("persistence.jl")

function __init__()
    _init_ffi!()
end

end # module ScienceKit
