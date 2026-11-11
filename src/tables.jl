function _column_map(table::NamedTuple)
    return Dict{Symbol, Any}( k => getfield(table, k) for k in keys(table) ), collect(Symbol, keys(table))
end

function _column_map(table::AbstractDict)
    ordered = Symbol[Symbol(k) for k in keys(table)]
    mapped = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in pairs(table))
    return mapped, ordered
end

function _column_map(rows::AbstractVector{<:NamedTuple})
    isempty(rows) && throw(ScienceKitError(STATUS_SHAPE, "cannot extract columns from an empty row table"))
    names_vec = collect(Symbol, keys(first(rows)))
    mapped = Dict{Symbol, Any}()
    for name in names_vec
        mapped[name] = [getfield(row, name) for row in rows]
    end
    return mapped, names_vec
end

function _column_map(table)
    names_raw = propertynames(table)
    isempty(names_raw) && throw(ScienceKitError(
        STATUS_INVALID_ARGUMENT,
        "table of type $(typeof(table)) exposes no column properties"
    ))
    names_vec = Symbol[Symbol(n) for n in names_raw]
    mapped = Dict{Symbol, Any}(name => getproperty(table, name) for name in names_vec)
    return mapped, names_vec
end

function _convert_numeric_column(col, col_name::Symbol, ::Type{T}, missing_policy)::Vector{T} where {T<:AbstractFloat}
    n = length(col)
    out = Vector{T}(undef, n)
    for (idx, val) in enumerate(col)
        if ismissing(val)
            if missing_policy === :reject
                throw(ScienceKitError(
                    STATUS_INVALID_ARGUMENT,
                    "column :$(col_name) contains `missing` at row $(idx) under missing_policy=:reject; " *
                    "pass missing_policy=:nan or missing_policy=(:fill => value)"
                ))
            elseif missing_policy === :nan
                out[idx] = T(NaN)
            elseif missing_policy isa Pair && first(missing_policy) === :fill
                out[idx] = T(last(missing_policy))
            else
                throw(ScienceKitError(
                    STATUS_INVALID_ARGUMENT,
                    "unknown missing_policy $(repr(missing_policy)); expected :reject, :nan, or (:fill => value)"
                ))
            end
        elseif val isa Real
            out[idx] = T(val)
        else
            throw(ScienceKitError(
                STATUS_INVALID_ARGUMENT,
                "column :$(col_name) contains non-numeric value of type $(typeof(val)); select numeric columns only"
            ))
        end
    end
    return out
end

"""
    extract_table(table; features=nothing, target=nothing, dtype=Float64, missing_policy=:reject)

Extract a dense feature `Matrix{dtype}` and optional target `Vector{dtype}` from
any column-accessible table (`NamedTuple` of vectors, `Vector{<:NamedTuple}`,
`Dict`, `DataFrame`, `Arrow.Table`, `CSV.File`, `TypedTables.Table`, or any
object exposing column properties).

Raises:
- `KeyError` when a requested feature or target column is absent
- `DimensionMismatch` when selected columns have uneven lengths
- `ScienceKitError(STATUS_INVALID_ARGUMENT, ...)` when `missing` appears under `:reject`
  or a selected column has non-numeric values
"""
function extract_table(
    table;
    features=nothing,
    target::Union{Symbol, AbstractString, Nothing}=nothing,
    dtype::Type=Float64,
    missing_policy=:reject,
)
    _validate_float_eltype(dtype, "table target dtype")
    cols_map, ordered_names = _column_map(table)

    target_sym = target === nothing ? nothing : Symbol(target)
    if target_sym !== nothing && !haskey(cols_map, target_sym)
        throw(KeyError(target_sym))
    end

    feature_syms = if features === nothing
        Symbol[name for name in ordered_names if name !== target_sym]
    else
        Symbol[Symbol(f) for f in features]
    end

    isempty(feature_syms) && throw(ScienceKitError(
        STATUS_INVALID_ARGUMENT,
        "at least one feature column must be selected"
    ))

    seen = Set{Symbol}()
    for sym in feature_syms
        !haskey(cols_map, sym) && throw(KeyError(sym))
        if sym in seen
            throw(ScienceKitError(STATUS_INVALID_ARGUMENT, "duplicate feature column :$(sym)"))
        end
        push!(seen, sym)
    end

    expected_rows = length(cols_map[first(feature_syms)])
    expected_rows <= 0 && throw(ScienceKitError(STATUS_SHAPE, "selected table columns are empty"))

    for sym in feature_syms
        actual = length(cols_map[sym])
        if actual != expected_rows
            throw(DimensionMismatch(
                "column :$(sym) has length $(actual), expected $(expected_rows) to match :$(first(feature_syms))"
            ))
        end
    end
    if target_sym !== nothing
        actual = length(cols_map[target_sym])
        if actual != expected_rows
            throw(DimensionMismatch(
                "target column :$(target_sym) has length $(actual), expected $(expected_rows)"
            ))
        end
    end

    X = Matrix{dtype}(undef, expected_rows, length(feature_syms))
    for (col_idx, sym) in enumerate(feature_syms)
        X[:, col_idx] = _convert_numeric_column(cols_map[sym], sym, dtype, missing_policy)
    end

    y = if target_sym === nothing
        nothing
    else
        _convert_numeric_column(cols_map[target_sym], target_sym, dtype, missing_policy)
    end

    return X, y
end
