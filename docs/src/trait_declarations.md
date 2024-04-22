# Trait declarations

Two trait functions allow the implementer to restrict the types of
data `X`, `y` and `Xnew` discussed above. The MLJ task interface uses
these traits for data type checks but also for model search. If they
are omitted (and your model is registered) then a general user may
attempt to use your model with inappropriately typed data.

The trait functions `input_scitype` and `target_scitype` take
scientific data types as values. We assume here familiarity with
[ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl)
(see [Getting Started](index.md) for the basics).

For example, to ensure that the `X` presented to the
`DecisionTreeClassifier` `fit` method is a table whose columns all
have `Continuous` element type (and hence `AbstractFloat` machine
type), one declares

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = MMI.Table(MMI.Continuous)
```

or, equivalently,

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = Table(Continuous)
```

If, instead, columns were allowed to have either: (i) a mixture of `Continuous` and `Missing`
values, or (ii) `Count` (i.e., integer) values, then the declaration would be

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = Table(Union{Continuous,Missing},Count)
```

Similarly, to ensure the target is an AbstractVector whose elements
have `Finite` scitype (and hence `CategoricalValue` machine type) we declare

```julia
MMI.target_scitype(::Type{<:DecisionTreeClassifier}) = AbstractVector{<:Finite}
```

## Multivariate targets

The above remarks continue to hold unchanged for the case multivariate
targets.  For example, if we declare

```julia
target_scitype(SomeSupervisedModel) = Table(Continuous)
```

then this constrains the target to be any table whose columns have `Continuous` element scitype (i.e., `AbstractFloat`), while

```julia
target_scitype(SomeSupervisedModel) = Table(Continuous, Finite{2})
```

restricts to tables with continuous or binary (ordered or unordered)
columns.

For predicting variable length sequences of, say, binary values
(`CategoricalValue`s) with some common size-two pool) we declare

```julia
target_scitype(SomeSupervisedModel) = AbstractVector{<:NTuple{<:Finite{2}}}
```

The trait functions controlling the form of data are summarized as follows:

method                   | return type       | declarable return values     | fallback value
-------------------------|-------------------|------------------------------|---------------
`input_scitype`          | `Type`            | some scientific type          | `Unknown`
`target_scitype`         | `Type`            | some scientific type         | `Unknown`


Additional trait functions tell MLJ's `@load` macro how to find your
model if it is registered, and provide other self-explanatory metadata
about the model:

method                       | return type       | declarable return values           | fallback value
-----------------------------|-------------------|------------------------------------|---------------
`load_path`                  | `String`          | unrestricted                       | "unknown"
`package_name`               | `String`          | unrestricted                       | "unknown"
`package_uuid`               | `String`          | unrestricted                       | "unknown"
`package_url`                | `String`          | unrestricted                       | "unknown"
`package_license`            | `String`          | unrestricted                       | "unknown"
`is_pure_julia`              | `Bool`            | `true` or `false`                  | `false`
`supports_weights`           | `Bool`            | `true` or `false`                  | `false`
`supports_class_weights`     | `Bool`            | `true` or `false`                  | `false`
`supports_training_losses`   | `Bool`            | `true` or `false`                  | `false`
`reports_feature_importances`| `Bool`            | `true` or `false`                  | `false`


Here is the complete list of trait function declarations for
`DecisionTreeClassifier`, whose core algorithms are provided by
DecisionTree.jl, but whose interface actually lives at
[MLJDecisionTreeInterface.jl](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl).

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:DecisionTreeClassifier}) = AbstractVector{<:MMI.Finite}
MMI.load_path(::Type{<:DecisionTreeClassifier}) = "MLJDecisionTreeInterface.DecisionTreeClassifier"
MMI.package_name(::Type{<:DecisionTreeClassifier}) = "DecisionTree"
MMI.package_uuid(::Type{<:DecisionTreeClassifier}) = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
MMI.package_url(::Type{<:DecisionTreeClassifier}) = "https://github.com/bensadeghi/DecisionTree.jl"
MMI.is_pure_julia(::Type{<:DecisionTreeClassifier}) = true
```

Alternatively, these traits can also be declared using `MMI.metadata_pkg` and `MMI.metadata_model` helper functions as:

```julia
MMI.metadata_pkg(
  DecisionTreeClassifier,
  name="DecisionTree",
  package_uuid="7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
  package_url="https://github.com/bensadeghi/DecisionTree.jl",
  is_pure_julia=true
)

MMI.metadata_model(
  DecisionTreeClassifier,
  input_scitype=MMI.Table(MMI.Continuous),
  target_scitype=AbstractVector{<:MMI.Finite},
  load_path="MLJDecisionTreeInterface.DecisionTreeClassifier"
)
```

*Important.* Do not omit the `load_path` specification. If unsure what
it should be, post an issue at
[MLJ](https://github.com/alan-turing-institute/MLJ.jl/issues).

```@docs
MMI.metadata_pkg
```

```@docs
MMI.metadata_model
```


