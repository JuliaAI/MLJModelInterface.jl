# Summary of methods

The compulsory and optional methods to be implemented for each concrete type
`SomeSupervisedModel <: MMI.Supervised` are summarized below.

An `=` indicates the return value for a fallback version of the
method.

Compulsory:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y) -> fitresult, cache, report
MMI.predict(model::SomeSupervisedModel, fitresult, Xnew) -> yhat
```

Optional, to check and correct invalid hyperparameter values:

```julia
MMI.clean!(model::SomeSupervisedModel) = ""
```

Optional, to return user-friendly form of fitted parameters:

```julia
MMI.fitted_params(model::SomeSupervisedModel, fitresult) = fitresult
```

Optional, to avoid redundant calculations when re-fitting machines
associated with a model:

```julia
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y) =
   MMI.fit(model, verbosity, X, y)
```

Optional, to specify default hyperparameter ranges (for use in tuning):

```julia
MMI.hyperparameter_ranges(T::Type) = Tuple(fill(nothing, length(fieldnames(T))))
```

Optional, if `SomeSupervisedModel <: Probabilistic`:

```julia
MMI.predict_mode(model::SomeSupervisedModel, fitresult, Xnew) =
	mode.(predict(model, fitresult, Xnew))
MMI.predict_mean(model::SomeSupervisedModel, fitresult, Xnew) =
	mean.(predict(model, fitresult, Xnew))
MMI.predict_median(model::SomeSupervisedModel, fitresult, Xnew) =
	median.(predict(model, fitresult, Xnew))
```

Required, if the model is to be registered (findable by general users):

```julia
MMI.load_path(::Type{<:SomeSupervisedModel})    = ""
MMI.package_name(::Type{<:SomeSupervisedModel}) = "Unknown"
MMI.package_uuid(::Type{<:SomeSupervisedModel}) = "Unknown"
```

```julia
MMI.input_scitype(::Type{<:SomeSupervisedModel}) = Unknown
```

Strongly recommended, to constrain the form of target data passed to fit:

```julia
MMI.target_scitype(::Type{<:SomeSupervisedModel}) = Unknown
```

Optional but recommended:

```julia
MMI.package_url(::Type{<:SomeSupervisedModel})  = "unknown"
MMI.is_pure_julia(::Type{<:SomeSupervisedModel}) = false
MMI.package_license(::Type{<:SomeSupervisedModel}) = "unknown"
```

If `SomeSupervisedModel` supports sample weights or class weights,
then instead of the `fit` above, one implements

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y, w=nothing) -> fitresult, cache, report
```

and, if appropriate

```julia
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y, w=nothing) =
   MMI.fit(model, verbosity, X, y, w)
```

Additionally, if `SomeSupervisedModel` supports sample weights, one must declare

```julia
MMI.supports_weights(model::Type{<:SomeSupervisedModel}) = true
```

Optionally, an implementation may add a data front-end, for transforming user data (such
as a table) into some model-specific format (such as a matrix), and/or add methods to
specify how reformatted data is resampled. **This alters the interpretation of the data
arguments of `fit`, `update` and `predict`, whose number may also change.** See
[Implementing a data front-end](@ref) for details). A data front-end provides the MLJ user
certain performance advantages when retraining a machine.

**Third-party packages that interact directly with models using the MLJModelInterface.jl
API, rather than through the machine interface, will also need to understand how the data
front-end works**, so they incorporate `reformat` into their `fit`/`update`/`predict`
calls. See also this
[issue](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl/issues/51).

```julia
MLJModelInterface.reformat(model::SomeSupervisedModel, args...) = args
MLJModelInterface.selectrows(model::SomeSupervisedModel, I, data...) = data
```

Optionally, to customized support for serialization of machines (see
[Serialization](@ref)), overload

```julia
MMI.save(filename, model::SomeModel, fitresult; kwargs...) = fitresult
```

and possibly

```julia
MMI.restore(filename, model::SomeModel, serializable_fitresult) -> serializable_fitresult
```

These last two are unlikely to be needed if wrapping pure Julia code.


