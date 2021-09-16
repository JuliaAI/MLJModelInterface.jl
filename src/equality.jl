# to ensure that propertynames that aren't fieldnames are always viewed as
# "defined", in MLJType objects. See #115.
function _isdefined(object, name)
    pnames = propertynames(object)
    fnames = fieldnames(typeof(object))
    name in pnames && !(name in fnames) && return true
    isdefined(object, name)
end

function _equal_to_depth_one(x1, x2)
    names = propertynames(x1)
    names === propertynames(x2) || return false
    for name in names
            getproperty(x1, name) == getproperty(x2, name) || return false
    end
    return true
end

@doc """
    deep_properties(::Type{<:MLJType})

Given an `MLJType` subtype `M`, the value of this trait should be a
tuple of any properties of `M` to be regarded as "deep".

When two instances of type `M` are to be tested for equality, in the
sense of `==` or `is_same_except`, then the values of a "deep"
property (whose values are assumed to be of composite type) are deemed
to agree if all corresponding properties *of those property values*
are `==`.

Any property of `M` whose values are themselves of `MLJType` are
"deep" automatically, and should not be included in the trait return
value.

See also [`is_same_except`](@ref)

### Example

Consider an `MLJType` subtype `Foo`, with a single field of
type `Bar` which is *not* a subtype of `MLJType`:

    mutable struct Bar
        x::Int
    end

    mutable struct Foo <: MLJType
        bar::Bar
    end

Then the mutability of `Foo` implies `Foo(1) != Foo(1)` and so, by the
definition `==` for `MLJType` objects (see [`is_same_except`](@ref))
we have

    Bar(Foo(1)) != Bar(Foo(1))

However after the declaration

    MLJModelInterface.deep_properties(::Type{<:Foo}) = (:bar,)

We have

    Bar(Foo(1)) == Bar(Foo(1))

"""
StatisticalTraits.deep_properties


"""
    is_same_except(m1, m2, exceptions::Symbol...; deep_properties=Symbol[])

If both `m1` and `m2` are of `MLJType`, return `true` if the
following conditions all hold, and `false` otherwise:

- `typeof(m1) === typeof(m2)`

- `propertynames(m1) === propertynames(m2)`

- with the exception of properties listed as `exceptions` or bound to
  an `AbstractRNG`, each pair of corresponding property values is
  either "equal" or both undefined. (If a property appears as a
  `propertyname` but not a `fieldname`, it is deemed as always defined.)

The meaining of "equal" depends on the type of the property value:

- values that are themselves of `MLJType` are "equal" if they are
equal in the sense of `is_same_except` with no exceptions.

- values that are not of `MLJType` are "equal" if they are `==`.

In the special case of a "deep" property, "equal" has a different
meaning; see [`deep_properties`](@ref)) for details.

If `m1` or `m2` are not `MLJType` objects, then return `==(m1, m2)`.

"""
is_same_except(x1, x2) = ==(x1, x2)
function is_same_except(m1::M1,
                        m2::M2,
                        exceptions::Symbol...) where {M1<:MLJType,M2<:MLJType}
    typeof(m1) === typeof(m2) || return false
    names = propertynames(m1)
    propertynames(m2) === names || return false

    for name in names
        if !(name in exceptions)
            if !_isdefined(m1, name)
               !_isdefined(m2, name) || return false
            elseif _isdefined(m2, name)
                if name in deep_properties(M1)
                    _equal_to_depth_one(getproperty(m1,name),
                                        getproperty(m2, name)) || return false
                else
                    (is_same_except(getproperty(m1, name),
                                    getproperty(m2, name)) ||
                     getproperty(m1, name) isa AbstractRNG ||
                     getproperty(m2, name) isa AbstractRNG) || return false
                end
            else
                return false
            end
        end
    end
    return true
end

==(m1::M1, m2::M2) where {M1<:MLJType,M2<:MLJType} = is_same_except(m1, m2)

# for using `replace` or `replace!` on collections of MLJType objects
# (eg, Model objects in a learning network) we need a stricter
# equality and a corresponding definition of `in`.
Base.isequal(m1::MLJType, m2::MLJType) = (m1 === m2)

# Note: To prevent julia crash, it seems we need to annotate the type
# of itr:
function special_in(x, itr)::Union{Bool,Missing}
    for y in itr
        ismissing(y) && return missing
        y === x && return true
    end
    return false
end
Base.in(x::MLJType, itr::Set) = special_in(x, itr)
Base.in(x::MLJType, itr::AbstractVector) = special_in(x, itr)
Base.in(x::MLJType, itr::Tuple) = special_in(x, itr)

# A version of `in` that actually uses `==`:

"""
    isrepresented(object::MLJType, objects)

Test if `object` has a representative in the iterable
`objects`. This is a weaker requirement than `object in objects`.

Here we say `m1` *respresents* `m2` if `is_same_except(m1, m2)` is
`true`.

"""
isrepresented(object::MLJType, ::Nothing) = false
function isrepresented(object::MLJType, itr)::Union{Bool,Missing}
    for m in itr
        ismissing(m) && return missing
        is_same_except(m, object) && return true
    end
    return false
end
