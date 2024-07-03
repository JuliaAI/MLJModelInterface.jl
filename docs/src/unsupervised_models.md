# Unsupervised models

Unsupervised models implement the MLJ model interface in a very
similar fashion. The main differences are:

- The `fit` method, which still returns `(fitresult, cache, report)` will typically have
  only one training argument `X`, as in `MLJModelInterface.fit(model, verbosity, X)`,
  although this is not a hard requirement; see [Transformers requiring a target variable
  in training](@ref) below.  Furthermore, in the case of models that subtype `Static <:
  Unsupervised` (see [Static models](@ref)) `fit` has no training arguments at all, but
  does not need to be implemented as a fallback returns `(nothing, nothing, nothing)`.

- A `transform` and/or `predict` method is implemented, and has the same signature as
  `predict` does in the supervised case, as in `MLJModelInterface.transform(model,
  fitresult, Xnew)`. However, it may only have one data argument `Xnew`, unless `model <:
  Static`, in which case there is no restriction.  A use-case for `predict` is K-means
  clustering that `predict`s labels and `transform`s input features into a space of lower
  dimension. See the [Transformers that also
  predict](https://JuliaAI.github.io/MLJ.jl/dev/transformers/#Transformers-that-also-predict)
  section of the MLJ manual for an example.

- The `target_scitype` refers to the output of `predict`, if implemented. A new trait,
  `output_scitype`, is for the output of `transform`. Unless the model is `Static` (see
  [Static models](@ref)) the trait `input_scitype` is for the single data argument
  of `transform` (and `predict`, if implemented). If `fit` has more than one data
  argument, you must overload the trait `fit_data_scitype`, which bounds the allowed
  `data` passed to `fit(model, verbosity, data...)` and will always be a `Tuple` type.

- An `inverse_transform` can be optionally implemented. The signature is the same as
  `transform`, as in `MLJModelInterface.inverse_transform(model::MyUnsupervisedModel,
  fitresult, Xout)`, which:
   - must make sense for any `Xout` for which `scitype(Xout) <:
     output_scitype(MyUnsupervisedModel)`; and
   - must return an object `Xin` satisfying `scitype(Xin) <:
     input_scitype(MyUnsupervisedModel)`.

For sample implementations, see MLJ's [built-in
transformers](https://github.com/JuliaAI/MLJModels.jl/blob/dev/src/builtins/Transformers.jl)
and the clustering models at
[MLJClusteringInterface.jl](https://github.com/jbrea/MLJClusteringInterface.jl).

## Transformers requiring a target variable in training

An `Unsupervised` model that is not `Static` may include a second argument `y` in it's
`fit` signature, as in `fit(::MyTransformer, verbosity, X, y)`. For example, some feature
selection tools require a target variable `y` in training. (Unlike `Supervised` models, an
`Unsupervised` model is not required to implement `predict`, and in pipelines it is the
output of `transform`, and not `predict`, that is always propagated to the next model.) Such a
model should overload the trait `target_in_fit`, as in this example:

```julia
MLJModelInterface.target_in_fit(::Type{<:MyTransformer}) = true
```

This ensures that such models can appear in pipelines, and that a target provided to the
pipeline model is passed on to the model in training. 

If the model implements more than one `fit` signature (e.g., one with a target `y` and one
without) then `fit_data_scitype` must also be overloaded, as in this example:

```julia
MLJModelInterface.fit_data_scitype(::Type{<:MyTransformer}) = Union{
    Tuple{Table(Continuous)},
	Tuple{Table(Continous), AbstractVector{<:Finite}},
}
```


