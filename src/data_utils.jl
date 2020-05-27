vtrait(X) = X |> trait |> Val

const REQUIRE = "(requires MLJBase to be loaded)"

errlight(s) = throw(InterfaceError("Only `MLJModelInterface` is loaded. " *
                                    "Import `MLJBase` in order to use `$s`."))

# ------------------------------------------------------------------------
# categorical, note: not exported to avoid clashes; this is fine because
# MLJBase loads CategoricalArrays and MLJ interfaces should use qualified
# statements.

categorical(a...; kw...) = categorical(get_interface_mode(), a...; kw...)

categorical(::LightInterface, a...; kw...) = errlight("categorical")

# ------------------------------------------------------------------------
# matrix

"""
    matrix(X; transpose=false)

If `X <: AbstractMatrix`, return `X` or `permutedims(X)` if `transpose=true`.
If `X` is a Tables.jl compatible table source, convert `X` into a `Matrix`
$REQUIRE.
"""
matrix(X; kw...) = matrix(get_interface_mode(), vtrait(X), X; kw...)

matrix(::Mode, ::Val{:other}, X::AbstractMatrix; transpose=false) =
    transpose ? permutedims(X) : X

matrix(::Mode, ::Val{:other}, X; kw...) =
    throw(ArgumentError("Function `matrix` only supports AbstractMatrix or " *
                        "containers implementing the Tables interface."))

matrix(::LightInterface, ::Val{:table}, X; kw...) = errlight("matrix")

# ------------------------------------------------------------------------
# int

"""
   int(x; type=nothing)

The positional integer of the `CategoricalString` or `CategoricalValue` `x`, in
the ordering defined by the pool of `x`. The type of `int(x)` is the reference
type of `x` $REQUIRE.

Not to be confused with `x.ref`, which is unchanged by reordering of the pool
of `x`, but has the same type.

    int(X::CategoricalArray)
    int(W::Array{<:CategoricalString})
    int(W::Array{<:CategoricalValue})

Broadcasted versions of `int`.

    julia> v = categorical([:c, :b, :c, :a])
    julia> levels(v)
    3-element Array{Symbol,1}:
     :a
     :b
     :c
    julia> int(v)
    4-element Array{UInt32,1}:
     0x00000003
     0x00000002
     0x00000003
     0x00000001

See also: [`decoder`](@ref).
"""
function int(x; type::Union{Nothing,Type{T}}=nothing) where T <: Real
    type === nothing && return int(get_interface_mode(), x)
    return convert.(T, int(get_interface_mode(), x))
end

int(::LightInterface, x) = errlight("int")

# ------------------------------------------------------------------------
# classes

"""
    classes(x)

All the categorical elements with the same pool as `x` (including `x`),
returned as a list, with an ordering consistent with the pool $REQUIRE.
Here `x` has `CategoricalValue` or `CategoricalString` type, and `classes(x)`
is a vector of the same eltype. Note that `x in classes(x)` is always true.

Not to be confused with `levels(x.pool)`. See the example below.

    julia>  v = categorical([:c, :b, :c, :a])
    4-element CategoricalArrays.CategoricalArray{Symbol,1,UInt32}:
     :c
     :b
     :c
     :a

    julia> levels(v)
    3-element Array{Symbol,1}:
     :a
     :b
     :c

    julia> x = v[4]
    CategoricalArrays.CategoricalValue{Symbol,UInt32} :a

    julia> classes(x)
    3-element CategoricalArrays.CategoricalArray{Symbol,1,UInt32}:
     :a
     :b
     :c

    julia> levels(x.pool)
    3-element Array{Symbol,1}:
     :a
     :b
     :c

"""
classes(x) = classes(get_interface_mode(), x)

classes(::LightInterface, x) = errlight("classes")

# ------------------------------------------------------------------------
# schema

"""
    schema(X)

If `X` is tabular, inspect the column types and scitypes, otherwise return
`nothing`.
"""
schema(X; kw...) = schema(get_interface_mode(), vtrait(X), X; kw...)

schema(::Mode, ::Val{:other}, X; kw...) = nothing

schema(::LightInterface, ::Val{:other}, X; kw...) = errlight("schema")

schema(::LightInterface, ::Val{:table}, X; kw...) = errlight("schema")

# ------------------------------------------------------------------------
# decoder

"""
    d = decoder(x)

A callable object for decoding the integer representation of a
`CategoricalString` or `CategoricalValue` sharing the same pool as `x`
$REQUIRE. (Here `x` is of one of these two types.) Specifically, one has
`d(int(y)) == y` for all `y in classes(x)`. One can also call `d` on integer
arrays, in which case `d` is broadcast over all elements.

    julia> v = categorical([:c, :b, :c, :a])
    julia> int(v)
    4-element Array{UInt32,1}:
     0x00000003
     0x00000002
     0x00000003
     0x00000001
    julia> d = decoder(v[3])
    julia> d(int(v)) == v
    true

*Warning:* It is *not* true that `int(d(u)) == u` always holds.

See also: [`int`](@ref), [`classes`](@ref).
"""
decoder(x) = decoder(get_interface_mode(), x)

decoder(::LightInterface, x) = errlight("decoder")

# ------------------------------------------------------------------------
# table

"""
    table(columntable; prototype=nothing)

Convert a named tuple of vectors or tuples `columntable`, into a table of the
"preferred sink type" of `prototype` $REQUIRE. This is often the type of
`prototype` itself, when `prototype` is a sink; see the Tables.jl
documentation. If `prototype` is not specified, then a named tuple of vectors
is returned.

    table(A::AbstractMatrix; names=nothing, prototype=nothing)

Wrap an abstract matrix `A` as a Tables.jl compatible table with the specified
column `names` (a tuple of symbols). If `names` are not specified,
`names=(:x1, :x2, ..., :xn)` is used, where `n=size(A, 2)` $REQUIRE.

If a `prototype` is specified, then the matrix is materialized as a table of
the preferred sink type of `prototype`, rather than wrapped. Note that if
`prototype` is *not* specified, then `matrix(table(A))` is essentially a no-op.
"""
table(X; kw...) = table(get_interface_mode(), X; kw...)

table(::LightInterface, X; kw...) = errlight("table")

# ------------------------------------------------------------------------
# nrows, select, selectrows, selectcols

"""
    nrows(X)

Return the number of rows for a table, abstract vector or matrix `X` $REQUIRE.
"""
nrows(X) = nrows(get_interface_mode(), vtrait(X), X)

nrows(::Mode, ::Val{:other}, X::AbstractVecOrMat) = size(X, 1)
nrows(::Mode, ::Val{:other}, X::Nothing) = 0

nrows(::Mode, ::Val{:other}, X) =
    throw(ArgumentError("Function `nrows` only supports AbstractVector or " *
                        "AbstractMatrix or containers implementing the " *
                        "Tables interface."))

nrows(::LightInterface, ::Val{:table}, X) = errlight("table")

"""
    selectrows(X, r)

Select single or multiple rows from a table, abstract vector or matrix `X`
$REQUIRE. If `X` is tabular, the object returned is a table of the
preferred sink type of `typeof(X)`, even if only a single row is selected.
"""
selectrows(X, r) = selectrows(get_interface_mode(), vtrait(X), X, r)

selectrows(::Mode, ::Val{:other}, ::Nothing, r) = nothing

selectrows(::Mode, ::Val{:other}, X::AbstractVector, r)          = X[r]
selectrows(::Mode, ::Val{:other}, X::AbstractVector, r::Integer) = X[r:r]
selectrows(::Mode, ::Val{:other}, X::AbstractVector, ::Colon)    = X

selectrows(::Mode, ::Val{:other}, X::AbstractMatrix, r)          = X[r, :]
selectrows(::Mode, ::Val{:other}, X::AbstractMatrix, r::Integer) = X[r:r, :]
selectrows(::Mode, ::Val{:other}, X::AbstractMatrix, ::Colon)    = X

selectrows(::Mode, ::Val{:other}, X, r) =
    throw(ArgumentError("Function `selectrows` only supports AbstractVector " *
                        "or AbstractMatrix or containers implementing the " * "Tables interface."))

selectrows(::LightInterface, ::Val{:table}, X, r; kw...) =
    errlight("selectrows")

"""
    selectcols(X, c)

Select single or multiple columns from a matrix or table `X` $REQUIRE. If `c`
is an abstract vector of integers or symbols, then the object returned
is a table of the preferred sink type of `typeof(X)`. If `c` is a
*single* integer or column, then an `AbstractVector` is returned.
"""
selectcols(X, c) = selectcols(get_interface_mode(), vtrait(X), X, c)

selectcols(::Mode, ::Val{:other}, ::Nothing, c) = nothing

selectcols(::Mode, ::Val{:other}, X::AbstractMatrix, r)       = X[:, r]
selectcols(::Mode, ::Val{:other}, X::AbstractMatrix, ::Colon) = X

selectcols(::Mode, ::Val{:other}, X, r) =
    throw(ArgumentError("Function `selectcols` only supports AbstractMatrix " *
                        "or containers implementing the Tables interface."))

selectcols(::LightInterface, ::Val{:table}, X, c; kw...) =
    errlight("selectcols")

"""
    select(X, r, c)

Select element(s) of a table or matrix at row(s) r and column(s) c. An object
of the sink type of X (or a matrix) is returned unless c is a single integer or
symbol. In that case a vector is returned, unless r is a single integer, in
which case a single element is returned.

See also: [`selectrows`](@ref), [`selectcols`](@ref).
"""
select(X, r, c) = select(get_interface_mode(), vtrait(X), X, r, c)

# only used here to denote "group of indices"
const MIdx = Union{AbstractArray,Colon}

select(::Mode, ::Val, X, r::MIdx, c)       = selectcols(selectrows(X, r), c)
select(::Mode, ::Val, X, r, c::MIdx)       = selectcols(selectrows(X, r), c)
select(::Mode, ::Val, X, r::MIdx, c::MIdx) = selectcols(selectrows(X, r), c)
select(::Mode, ::Val, X, r, c) = _squeeze(selectcols(selectrows(X, r), c))

_squeeze(::Nothing) = nothing
_squeeze(v) = first(v)

# ------------------------------------------------------------------------
# UnivariateFinite

const UNIVARIATE_FINITE_DOCSTRING =
"""
    UnivariateFinite(classes, p; pool=nothing, ordered=false)

Construct a discrete univariate distribution whose finite support is
the elements of the vector `classes`, and whose corresponding
probabilities are elements of the vector `p`, which must sum to one $REQUIRE.

*Important.* Here `classes` must have type
 `AbstractVector{<:CategoricalValue}` and all elements are assumed to
share the same categorical pool. Raw classes *may* be used, but only provided
`pool` is specified. The possible values are:

- some `v::CategoricalVector`  such that `classes` is a subset of `levels(v)`

- some `a::CategoricalValue` such that `classes` is a subset of `levels(a)`

- `missing`, in which case a new categorical pool is created which has
  `classes` as its only levels.

In the last case specify `ordered=true` to order the new pool.

        UnivariateFinite(prob_given_class; pool=nothing, ordered=false)

Construct a discrete univariate distribution whose finite support is
the set of keys of the provided dictionary, `prob_given_class`, and
whose values specify the corresponding probabilities $REQUIRE.

The type requirements on the keys of the dictionary are the same as
`classes` above.

"""
UnivariateFinite(d::AbstractDict; kwargs...) =
    UnivariateFinite(get_interface_mode(), d; kwargs...)
UnivariateFinite(c::AbstractVector, p; kwargs...) =
    UnivariateFinite(get_interface_mode(), c, p; kwargs...)

UnivariateFinite(::LightInterface, a...; kwargs...) =
    errlight("UnivariateFinite")

const UNIVARIATE_FINITE_VECTOR_DOCSTRING =
"""
    UnivariateFiniteVector(scores, classes)

Container for UnivariateFinite elements optimised for efficiency.
Accessing a single element will construct and return the corresponding
UnivariateFinite lazily.
"""
UnivariateFiniteVector(s::AbstractArray) =
    UnivariateFiniteVector(get_interface_mode(), s)
UnivariateFiniteVector(s::AbstractArray, c) =
    UnivariateFiniteVector(get_interface_mode(), s, c)

UnivariateFiniteVector(::LightInterface, a...) =
    errlight("UnivariateFiniteVector")
