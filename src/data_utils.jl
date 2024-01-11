
const REQUIRE = "(requires MLJBase to be loaded)"

function errlight(s)
    throw(
        InterfaceError(
            "Only `MLJModelInterface` is loaded. " *
            "Import `MLJBase` in order to use `$s`."
        )
    )
end

## Internal function to be extended in MLJBase (so do not export)                                   
vtrait(X, s="") = vtrait(get_interface_mode(), X, s)
vtrait(::LightInterface, X, s) = errlight(s)

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

If `X isa AbstractMatrix`, return `X` or `permutedims(X)` if `transpose=true`.
Otherwise if `X` is a Tables.jl compatible table source, convert `X` into a `Matrix`.

"""
function matrix(X; kw...) 
    m = get_interface_mode()
    return matrix(m, vtrait(m, X, "matrix"), X; kw...)
end

function matrix(X::AbstractMatrix; transpose=false)
    return transpose ? permutedims(X) : X
end

matrix(m::Mode, v, X; kw...) = _matrix(m, v, X; kw...)
matrix(::LightInterface, v, X; kw...) = errlight("matrix")

function _matrix(::Mode, ::Val{:other}, X; kw...)
    throw(
        ArgumentError(
            "`matrix` method only supports AbstractMatrix or " *
            "containers implementing the Tables interface."
        )
    )
end

# ------------------------------------------------------------------------
# int

"""
   int(x)

The positional integer of the `CategoricalString` or `CategoricalValue` `x`, in
the ordering defined by the pool of `x`. The type of `int(x)` is the reference
type of `x`.

Not to be confused with `x.ref`, which is unchanged by reordering of the pool
of `x`, but has the same type.

    int(X::CategoricalArray)
    int(W::Array{<:CategoricalString})
    int(W::Array{<:CategoricalValue})

Broadcasted versions of `int`.

```julia
julia> v = categorical(["c", "b", "c", "a"])
4-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "c"
 "b"
 "c"
 "a"

julia> levels(v)
3-element Vector{String}:
 "a"
 "b"
 "c"

julia> int(v)
4-element Vector{UInt32}:
 0x00000003
 0x00000002
 0x00000003
 0x00000001
```
See also: [`decoder`](@ref).
"""
function int(x; type=nothing)
    type === nothing && return int(get_interface_mode(), x)
    return convert.(type, int(get_interface_mode(), x))
end

int(::LightInterface, x) = errlight("int")

# ------------------------------------------------------------------------
# classes

"""
    classes(x)

All the categorical elements with the same pool as `x` (including
`x`), returned as a list, with an ordering consistent with the pool.
Here `x` has `CategoricalValue` type, and
`classes(x)` is a vector of the same eltype. Note that `x in
classes(x)` is always true.

Not to be confused with `levels(x.pool)`. See the example below.

```julia
julia>  v = categorical(["c", "b", "c", "a"])
4-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "c"
 "b"
 "c"
 "a"

julia> levels(v)
3-element Vector{String}:
 "a"
 "b"
 "c"

julia> x = v[4]
CategoricalArrays.CategoricalValue{String, UInt32} "a"

julia> classes(x)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "a"
 "b"
 "c"

julia> levels(x.pool)
3-element Vector{String}:
 "a"
 "b"
 "c"
```
"""
classes(x) = classes(get_interface_mode(), x)

classes(::LightInterface, x) = errlight("classes")

# ------------------------------------------------------------------------
# scitype

"""
    scitype(X)

The scientific type (interpretation) of `X`, distinct from its
machine type.

### Examples
```julia
julia> scitype(3.14)
Continuous

julia> scitype([1, 2, missing])
AbstractVector{Union{Missing, Count}} 

julia> scitype((5, "beige"))
Tuple{Count, Textual}

julia> using CategoricalArrays

julia> X = (gender = categorical(['M', 'M', 'F', 'M', 'F']),
        ndevices = [1, 3, 2, 3, 2]);

julia> scitype(X)
Table{Union{AbstractVector{Count}, AbstractVector{Multiclass{2}}}}
```
"""
scitype(X) = scitype(get_interface_mode(), vtrait(X, "scitype"), X)

function scitype(::LightInterface, m, X)
    return errlight("scitype")
end

# ------------------------------------------------------------------------
# schema

"""
    schema(X)

Inspect the column types and scitypes of a tabular object.
returns `nothing` if the column types and scitypes can't be inspected.
"""
schema(X) = schema(get_interface_mode(), vtrait(X, "schema"), X)

function schema(::LightInterface, m, X)
    return errlight("schema")
end

# ------------------------------------------------------------------------
# istable

"""
    istable(X)

Return true if `X` is tabular.
"""
function istable(X)
    m = get_interface_mode()
    return istable(m, vtrait(m, X, "istable"))
end

istable(::Mode, ::Val{:other}) = false

istable(::Mode, ::Val{:table}) = true

# ------------------------------------------------------------------------
# decoder

"""
    decoder(x)

Return a callable object for decoding the integer representation of a
`CategoricalValue` sharing the same pool the `CategoricalValue`
`x`. Specifically, one has `decoder(x)(int(y)) == y` for all
`CategoricalValue`s `y` having the same pool as `x`. One can also call
`decoder(x)` on integer arrays, in which case `decoder(x)` is
broadcast over all elements.

### Examples
```julia
julia> v = categorical(["c", "b", "c", "a"])
4-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "c"
 "b"
 "c"
 "a"

julia> int(v)
4-element Vector{UInt32}:
 0x00000003
 0x00000002
 0x00000003
 0x00000001

julia> d = decoder(v[3]);

julia> d(int(v)) == v
true
```
### Warning:

It is *not* true that `int(d(u)) == u` always holds.

See also: [`int`](@ref).
"""
decoder(x) = decoder(get_interface_mode(), x)

decoder(::LightInterface, x) = errlight("decoder")

# ------------------------------------------------------------------------
# table

"""
    table(columntable; prototype=nothing)

Convert a named tuple of vectors or tuples `columntable`, into a table
of the "preferred sink type" of `prototype`. This is often the type of
`prototype` itself, when `prototype` is a sink; see the Tables.jl
documentation. If `prototype` is not specified, then a named tuple of
vectors is returned.

    table(A::AbstractMatrix; names=nothing, prototype=nothing)

Wrap an abstract matrix `A` as a Tables.jl compatible table with the
specified column `names` (a tuple of symbols). If `names` are not
specified, `names=(:x1, :x2, ..., :xn)` is used, where `n=size(A, 2)`.

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

Return the number of rows for a table, `AbstractVector` or `AbtractMatrix`, `X`.
"""
function nrows(X)
    m = get_interface_mode()
    return nrows(m, vtrait(m, X, "nrows"), X)
end

nrows(::Nothing) = 0

nrows(m::Mode, v, X) = _nrows(m, v, X)
nrows(::LightInterface, v, X) = errlight("table")

_nrows(::Mode, ::Val{:other}, X::AbstractVecOrMat) = size(X, 1)
_nrows(::Mode, ::Val{:other}, X::Nothing) = 0

function _nrows(::Mode, ::Val{:other}, X)
    throw(
        ArgumentError(
            "`nrows` method only supports AbstractVector or " *
            "AbstractMatrix or containers implementing the " *
            "Tables interface."
        )
    )
end

nrows(::LightInterface, ::Val{:table}, X) = errlight("table")

"""
    selectrows(X, r)

Select single or multiple rows from a table, abstract vector or matrix
`X`. If `X` is tabular, the object returned is a table of the
preferred sink type of `typeof(X)`, even if only a single row is
selected.

If the object is neither a table, abstract vector or matrix, `X` is
returned and `r` is ignored.

"""
function selectrows end

function selectrows(X, r)
    m = get_interface_mode()
    return selectrows(m, vtrait(m, X, "selectrows"), X, r)
end

selectrows(::Nothing, r) = nothing
selectrows(m::Mode, v, X, r) = _selectrows(m, v, X, r)
selectrows(::LightInterface, v, X, r) = errlight("selectrows")

# fall-back is to return object, ignoring vector of row indices, `r`:
_selectrows(::Mode, ::Val{:other}, X::Any, r) = X

_selectrows(::Mode, ::Val{:other}, X::AbstractVector, r) = X[r]
_selectrows(::Mode, ::Val{:other}, X::AbstractVector, r::Integer) = X[r:r]
_selectrows(::Mode, ::Val{:other}, X::AbstractVector, ::Colon) = X

_selectrows(::Mode, ::Val{:other}, X::AbstractMatrix, r) = X[r, :]
_selectrows(::Mode, ::Val{:other}, X::AbstractMatrix, r::Integer) = X[r:r, :]
_selectrows(::Mode, ::Val{:other}, X::AbstractMatrix, ::Colon) = X

# The following method maybe called by `select`
_selectrows(::Mode, ::Val{:other}, ::Nothing, r) = nothing

"""
    selectcols(X, c)

Select single or multiple columns from a matrix or table `X`. If `c`
is an abstract vector of integers or symbols, then the object returned
is a table of the preferred sink type of `typeof(X)`. If `c` is a
*single* integer or column, then an `AbstractVector` is returned.

"""
function selectcols end

function selectcols(X, c)
    m = get_interface_mode()
    return selectcols(m, vtrait(m, X, "selectcols"), X, c)
end

selectcols(::Nothing, c) = nothing
selectcols(m::Mode, v, X, r) = _selectcols(m, v, X, r)
selectcols(::LightInterface, v, X, c) = errlight("selectcols")

_selectcols(::Mode, ::Val{:other}, X::AbstractMatrix, r) = X[:, r]
_selectcols(::Mode, ::Val{:other}, X::AbstractMatrix, ::Colon) = X
_selectcols(::Mode, ::Val{:other}, ::Nothing, r) = nothing

function _selectcols(::Mode, ::Val{:other}, X, r)
    throw(
        ArgumentError(
            "`selectcols` method only supports AbstractMatrix " *
            "or containers implementing the Tables interface."
        )
    )
end

"""
    select(X, r, c)

Select element(s) of a table or matrix at row(s) `r` and column(s) `c`. An object
of the sink type of `X` (or a matrix) is returned unless `c` is a single integer or
symbol. In that case a vector is returned, unless `r` is a single integer, in
which case a single element is returned.

See also: [`selectrows`](@ref), [`selectcols`](@ref).
"""
function select end

function select(X, r, c) 
    m = get_interface_mode()
    return select(m, vtrait(m, X, "select"), X, r, c)
end

select(::Nothing, r, c) = nothing

# only used here to denote "group of indices"

const MIdx = Union{AbstractArray, Colon}
select(m::Mode, v::Val, X, r, c) = _select(m, v, X, r, c)
select(m::LightInterface, v::Val, X, r, c) = errlight("select")

_select(m, v::Val{:table}, X, r, c) = __select(m, v, X, r, c)
_select(m, v::Val{:other}, X::AbstractMatrix, r, c) = __select(m, v, X, r, c)

__select(m::Mode, v::Val, X, r::MIdx, c) = selectcols(m, v, selectrows(m, v, X, r), c)
__select(m::Mode, v::Val, X, r, c::MIdx) = selectcols(m, v, selectrows(m, v, X, r), c)
__select(m::Mode, v::Val, X, r::MIdx, c::MIdx) = selectcols(m, v, selectrows(m, v, X, r), c)
__select(m::Mode, v::Val, X, r, c) = _squeeze(selectcols(m, v, selectrows(m, v, X, r), c))

function _select(m, ::Val{:other}, X, r, c)
    throw(
        ArgumentError(
            "`select` method only supports AbstractMatrix " *
            "or containers implementing the Tables interface."
        )
    )
end

_squeeze(::Nothing) = nothing
_squeeze(v) = first(v)

# ------------------------------------------------------------------------
# UnivariateFinite

const UNIVARIATE_FINITE_DOCSTRING =
    """
        UnivariateFinite(
            support,
            probs;
            pool=nothing,
            augmented=false,
            ordered=false
        )

    Construct a discrete univariate distribution whose finite support is
    the elements of the vector `support`, and whose corresponding
    probabilities are elements of the vector `probs`. Alternatively,
    construct an abstract *array* of `UnivariateFinite` distributions by
    choosing `probs` to be an array of one higher dimension than the array
    generated.

    Here the word "probabilities" is an abuse of terminology as there is
    no requirement that probabilities actually sum to one, only that they
    be non-negative. So `UnivariateFinite` objects actually implement
    arbitrary non-negative measures over finite sets of labelled points. A
    `UnivariateDistribution` will be a bona fide probability measure when
    constructed using the `augment=true` option (see below) or when
    `fit` to data.

    Unless `pool` is specified, `support` should have type
    `AbstractVector{<:CategoricalValue}` and all elements are assumed to
    share the same categorical pool, which may be larger than `support`.

    *Important.* All levels of the common pool have associated
    probabilities, not just those in the specified `support`. However,
    these probabilities are always zero (see example below).

    If `probs` is a matrix, it should have a column for each class in
    `support` (or one less, if `augment=true`). More generally, `probs`
    will be an array whose size is of the form `(n1, n2, ..., nk, c)`,
    where `c = length(support)` (or one less, if `augment=true`) and the
    constructor then returns an array of `UnivariateFinite` distributions
    of size `(n1, n2, ..., nk)`.

    ## Examples

    ```julia
    julia> v = categorical(["x", "x", "y", "x", "z"])
    5-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
     "x"
     "x"
     "y"
     "x"
     "z"

    julia> UnivariateFinite(classes(v), [0.2, 0.3, 0.5])
    UnivariateFinite{Multiclass{3}}(x=>0.2, y=>0.3, z=>0.5)

    julia> d = UnivariateFinite([v[1], v[end]], [0.1, 0.9])
    UnivariateFinite{Multiclass{3}}(x=>0.1, z=>0.9)

    julia> rand(d, 3)
    3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
     "x"
     "z"
     "x"

    julia> levels(d)
    3-element Vector{String}:
     "x"
     "y"
     "z"

    julia> pdf(d, "y")
    0.0

    ```

    ### Specifying a pool

    Alternatively, `support` may be a list of raw (non-categorical)
    elements if `pool` is:

    - some `CategoricalArray`, `CategoricalValue` or `CategoricalPool`,
    such that `support` is a subset of `levels(pool)`

    - `missing`, in which case a new categorical pool is created which has
    `support` as its only levels.

    In the last case, specify `ordered=true` if the pool is to be
    considered ordered.

    ```julia
    julia> UnivariateFinite(["x", "z"], [0.1, 0.9], pool=missing, ordered=true)
    UnivariateFinite{OrderedFactor{2}}(x=>0.1, z=>0.9)

    julia> d = UnivariateFinite(["x", "z"], [0.1, 0.9], pool=v) # v defined above
    UnivariateFinite{Multiclass{3}}(x=>0.1, z=>0.9)

    julia> pdf(d, "y") # allowed as `"y" in levels(v)`
    0.0

    julia> v = categorical(["x", "x", "y", "x", "z", "w"])
    6-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
     "x"
     "x"
     "y"
     "x"
     "z"
     "w"

    julia> probs = rand(100, 3); probs = probs ./ sum(probs, dims=2);

    julia> UnivariateFinite(["x", "y", "z"], probs, pool=v)
    100-element UnivariateFiniteVector{Multiclass{4}, String, UInt32, Float64}:
     UnivariateFinite{Multiclass{4}}(x=>0.194, y=>0.3, z=>0.505)
     UnivariateFinite{Multiclass{4}}(x=>0.727, y=>0.234, z=>0.0391)
     UnivariateFinite{Multiclass{4}}(x=>0.674, y=>0.00535, z=>0.321)
     ⋮
     UnivariateFinite{Multiclass{4}}(x=>0.292, y=>0.339, z=>0.369)
    ```

    ### Probability augmentation

    If `augment=true` the provided array is augmented by inserting
    appropriate elements *ahead* of those provided, along the last
    dimension of the array. This means the user only provides probabilities
    for the classes `c2, c3, ..., cn`. The class `c1` probabilities are
    chosen so that each `UnivariateFinite` distribution in the returned
    array is a bona fide probability distribution.

    ---

        UnivariateFinite(prob_given_class; pool=nothing, ordered=false)

    Construct a discrete univariate distribution whose finite support is
    the set of keys of the provided dictionary, `prob_given_class`, and
    whose values specify the corresponding probabilities.

    The type requirements on the keys of the dictionary are the same as
    the elements of `support` given above with this exception: if
    non-categorical elements (raw labels) are used as keys, then
    `pool=...` must be specified and cannot be `missing`.

    If the values (probabilities) are arrays instead of scalars, then an
    abstract array of `UnivariateFinite` elements is created, with the
    same size as the array.

    """

@doc UNIVARIATE_FINITE_DOCSTRING
function UnivariateFinite(d::AbstractDict; kwargs...)
    return UnivariateFinite(get_interface_mode(), d; kwargs...)
end

function UnivariateFinite(support::AbstractVector, probs; kwargs...)
    return UnivariateFinite(get_interface_mode(), support, probs; kwargs...)
end

function UnivariateFinite(probs; kwargs...)
    return UnivariateFinite(get_interface_mode(), probs; kwargs...)
end

function UnivariateFinite(::LightInterface, a...; kwargs...)
    return errlight("UnivariateFinite")
end

