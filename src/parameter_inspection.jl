istransparent(::Any) = false
istransparent(::MLJType) = true

"""
    params(m::MLJType)

Recursively convert any transparent object `m` into a named tuple,
keyed on the fields of `m`. An object is *transparent* if
`MLJModelInterface.istransparent(m) == true`. The named tuple is
possibly nested because `params` is recursively applied to the field
values, which themselves might be transparent.

Most objects of type `MLJType` are transparent.

```julia
julia> params(EnsembleModel(atom=ConstantClassifier()))
(atom = (target_type = Bool,),
weights = Float64[],
bagging_fraction = 0.8,
rng_seed = 0,
n = 100,
parallel = true,)
```
"""
params(m) = params(m, Val(istransparent(m)))
params(m, ::Val{false}) = m

function params(m, ::Val{true})
    fields = fieldnames(typeof(m))
    return NamedTuple{fields}(Tuple([params(getfield(m, field)) for field in fields]))
end

isamodel(::Any) = false
isamodel(::Model) = true

"""
    flat_params(m::Model)

Recursively convert any object subtyping `Model` into a named tuple, keyed on
the property names of `m`. The named tuple is possibly nested because
`flat_params` is recursively applied to the property values, which themselves
might subtype `Model`.

For most `Model` objects, properties are synonymous with fields, but this is
not a hard requirement.

    julia> flat_params(EnsembleModel(atom=ConstantClassifier()))
    (atom = (target_type = Bool,),
     weights = Float64[],
     bagging_fraction = 0.8,
     rng_seed = 0,
     n = 100,
     parallel = true,)

"""
flat_params(m; prefix="") = flat_params(m, Val(isamodel(m)); prefix=prefix)
flat_params(m, ::Val{false}; prefix="") = NamedTuple{(Symbol(prefix),), Tuple{Any}}((m,))
function flat_params(m, ::Val{true}; prefix="")
    fields = propertynames(m)
    if isempty(fields)
        return NamedTuple{(Symbol(prefix),)}((m,))
    end
    prefix = prefix == "" ? "" : prefix * "__"
    merge([flat_params(getproperty(m, field); prefix="$(prefix)$(field)") for field in fields]...)
end
