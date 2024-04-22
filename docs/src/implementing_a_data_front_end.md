# Implementing a data front-end

!!! note

	It is suggested that packages implementing MLJ's model API, that later implement a data front-end, should tag their changes in a breaking release. (The changes will not break the use of models for the ordinary MLJ user, who interacts with models exclusively through the machine interface. However, it will break usage for some external packages that have chosen to depend directly on the model API.)

```julia
MLJModelInterface.reformat(model, args...) -> data
MLJModelInterface.selectrows(::Model, I, data...) -> sampled_data
```

Models optionally overload `reformat` to define transformations of
user-supplied data into some model-specific representation (e.g., from
a table to a matrix). Computational overheads associated with multiple
`fit!`/`predict`/`transform` calls (on MLJ machines) are then avoided
when memory resources allow. The fallback returns `args` (no
transformation).

The `selectrows(::Model, I, data...)` method is overloaded to specify
how the model-specific data is to be subsampled, for some observation
indices `I` (a colon, `:`, or instance of
`AbstractVector{<:Integer}`). In this way, implementing a data
front-end also allows more efficient resampling of data (in user calls
to `evaluate!`).

After detailing formal requirements for implementing a data front-end, we give a [Sample
implementation](@ref). A simple
[implementation](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl/blob/7e39bac6bce6d1736e4974f984b6e12801191dd5/src/MLJDecisionTreeInterface.jl#L453)
also appears in the MLJDecisionTreeInterface.jl package.

Here "user-supplied data" is what the MLJ user supplies when
constructing a machine, as in `machine(models, args...)`, which
coincides with the arguments expected by `fit(model, verbosity,
args...)` when `reformat` is not overloaded.

Overloading `reformat` is permitted for any `Model`
subtype, except for subtypes of `Static`. Here is a complete list of
responsibilities for such an implementation, for some
`model::SomeModelType` (a sample implementation follows after):

- A `reformat(model::SomeModelType, args...) -> data` method must be
  implemented for each form of `args...` appearing in a valid machine
  construction `machine(model, args...)` (there will be one for each
  possible signature of `fit(::SomeModelType, ...)`).

- Additionally, if not included above, there must be a single argument
  form of reformat, `reformat(model::SomeModelType, arg) -> (data,)`,
  serving as a data front-end for operations like `predict`. It must
  always hold that `reformat(model, args...)[1] = reformat(model,
  args[1])`.

The fallback is `reformat(model, args...) = args` (i.e., slurps provided data).

*Important.* `reformat(model::SomeModelType, args...)` must always return a tuple, even if
  this has length one. The length of the tuple need not match `length(args)`.
- `fit(model::SomeModelType, verbosity, data...)` should be
  implemented as if `data` is the output of `reformat(model,
  args...)`, where `args` is the data an MLJ user has bound to `model`
  in some machine. The same applies to any overloading of `update`.

- Each implemented operation, such as `predict` and `transform` - but
  excluding `inverse_transform` - must be defined as if its data
  arguments are `reformat`ed versions of user-supplied data. For
  example, in the supervised case, `data_new` in
  `predict(model::SomeModelType, fitresult, data_new)` is
  `reformat(model, Xnew)`, where `Xnew` is the data provided by the MLJ
  user in a call `predict(mach, Xnew)` (`mach.model == model`).

- To specify how the model-specific representation of data is to be
  resampled, implement `selectrows(model::SomeModelType, I, data...)
  -> resampled_data` for each overloading of `reformat(model::SomeModel,
  args...) -> data` above. Here `I` is an arbitrary abstract integer
  vector or `:` (type `Colon`).

*Important.* `selectrows(model::SomeModelType, I, args...)` must always
return a tuple of the same length as `args`, even if this is one.

The fallback for `selectrows` is described at [`selectrows`](@ref).


## Sample implementation

Suppose a supervised model type `SomeSupervised` supports sample
weights, leading to two different `fit` signatures, and that it has a
single operation `predict`:

```julia
fit(model::SomeSupervised, verbosity, X, y)
fit(model::SomeSupervised, verbosity, X, y, w)

predict(model::SomeSupervised, fitresult, Xnew)
```

Without a data front-end implemented, suppose `X` is expected to be a
table and `y` a vector, but suppose the core algorithm always converts
`X` to a matrix with features as rows (each record corresponds to
a column in the table).  Then a new data-front end might look like
this:

```julia
constant MMI = MLJModelInterface

# for fit:
MMI.reformat(::SomeSupervised, X, y) = (MMI.matrix(X)', y)
MMI.reformat(::SomeSupervised, X, y, w) = (MMI.matrix(X)', y, w)
MMI.selectrows(::SomeSupervised, I, Xmatrix, y) =
        (view(Xmatrix, :, I), view(y, I))
MMI.selectrows(::SomeSupervised, I, Xmatrix, y, w) =
        (view(Xmatrix, :, I), view(y, I), view(w, I))

# for predict:
MMI.reformat(::SomeSupervised, X) = (MMI.matrix(X)',)
MMI.selectrows(::SomeSupervised, I, Xmatrix) = (view(Xmatrix, :, I),)
```

With these additions, `fit` and `predict` are refactored, so that `X`
and `Xnew` represent matrices with features as rows.
