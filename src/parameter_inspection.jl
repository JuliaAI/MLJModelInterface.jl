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

```julia-repl
julia> params(EnsembleModel(model=ConstantClassifier()))
(model = (target_type = Bool,),
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

isnotaleaf(::Any) = false
isnotaleaf(m::Model) = length(propertynames(m)) > 0

"""
    flat_params(m::Model)

Deconstruct any `Model` instance `model` as a flat named tuple, keyed on property
names. Properties of nested model instances are recursively exposed,.as shown in the
example below.  For most `Model` objects, properties are synonymous with fields, but this
is not a hard requirement.

```julia-repl
julia> using MLJModels
julia> using EnsembleModels
julia> tree = (@load DecisionTreeClassifier pkg=DecisionTree)();

julia> flat_params(EnsembleModel(model=tree))
(model__max_depth = -1,
 model__min_samples_leaf = 1,
 model__min_samples_split = 2,
 model__min_purity_increase = 0.0,
 model__n_subfeatures = 0,
 model__post_prune = false,
 model__merge_purity_threshold = 1.0,
 model__display_depth = 5,
 model__feature_importance = :impurity,
 model__rng = Random._GLOBAL_RNG(),
 atomic_weights = Float64[],
 bagging_fraction = 0.8,
 rng = Random._GLOBAL_RNG(),
 n = 100,
 acceleration = CPU1{Nothing}(nothing),
 out_of_bag_measure = Any[],)
```


"""
flat_params(m; prefix="") = flat_params(m, Val(isnotaleaf(m)); prefix=prefix)
function flat_params(m, ::Val{false}; prefix="")
    prefix == "" && return NamedTuple()
    NamedTuple{(Symbol(prefix),), Tuple{Any}}((m,))
end
function flat_params(m, ::Val{true}; prefix="")
    fields = propertynames(m)
    prefix = prefix == "" ? "" : prefix * "__"
    merge([flat_params(getproperty(m, field); prefix="$(prefix)$(field)") for field in fields]...)
end
