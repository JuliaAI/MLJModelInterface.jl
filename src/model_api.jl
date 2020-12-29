"""
    fit(model, verbosity, data...) -> fitresult, cache, report

All models must implement a `fit` method. Here `data` is the
output of `reformat` on user-provided data, or some some resampling
thereof. The fallback of `reformat` returns the user-provided data
(eg, a table).

"""
function fit end

# fallback for static transformations
fit(::Static, ::Integer, data...) = (nothing, nothing, nothing)

# fallbacks for supervised models that don't support sample weights:
fit(m::Supervised, verbosity, X, y, w) = fit(m, verbosity, X, y)

"""
    update(model, verbosity, fitresult, cache, data...)

Models may optionally implement an `update` method. The fallback calls
`fit`.

"""
update(m::Model, verbosity, fitresult, cache, data...) =
    fit(m, verbosity, data...)

# to support online learning in the future:
# https://github.com/alan-turing-institute/MLJ.jl/issues/60 :
function update_data end

"""
    MLJModelInterface.reformat(model, args...) -> data

Models optionally overload `reformat` to define transformations of
user-supplied data into some model-specific representation (e.g., from
a table to a matrix). Computational overheads associated with multiple
`fit!`/`predict`/`transform` calls are then avoided, when memory
resources allow. The fallback returns `args` (no transformation).

Here "user-supplied data" is what the MLJ user supplies when
constructing a machine, as in `machine(models, args...)`, which
coincides with the arguments expected by `fit(model, verbosity,
args...)` when `reformat` is not overloaded.

Implementing a `reformat` data front-end is permitted for any `Model`
subtype, except for subtypes of `Static`. Here is a complete list of
responsibilities for such an implementation, for some
`model::SomeModelType`:

- A `reformat(model::SomeModelType, args...) -> data` method must be
  implemented for each form of `args...` appearing in a valid machine
  construction `machine(model, args...)` (there will be one for each
  possible signature of `fit(::SomeModelType, ...)`).

- Additionally, if not included above, there must be a single argument
  form of reformat, `reformat(model::SommeModelType, arg) -> (data,)`,
  serving as a data front-end for operations like `predict`. It must
  always hold that `reformat(model, args...)[1] = reformat(model,
  args[1])`.

**Warning.** `reformat(model::SomeModelType, args...)` must always
  return a tuple of the same length as `args`, even if this is one.

- `fit(model::SomeModelType, verbosity, data...)` should be
  implemented as if `data` is the output of `reformat(model,
  args...)`, where `args` is the data an MLJ user has bound to `model`
  in some machine. The same applies to any overloading of `update`.

- Each implemented operation, such as `predict` and `transform` - but
  excluding `inverse_transform` - must be defined as if its data
  arguments are `reformat`ed versions of user-supplied data. For
  example, in the supervised case, `data_new` in
  `predict(model::SomeModelType, fitresult, data_new)` is
  `reformat(model, Xnew)`, where `Xnew is the data provided by the MLJ
  user in a call `predict(mach, Xnew)` (`mach.model == model`).

- To specify how the model-specific representation of data is to be
  resampled, implement `selectrows(model::SomeModelType, I, data...)
  -> resampled_data` for each overloading of `reformat(model::SomeModel,
  args...) -> data` above. Here `I` is an arbitrary abstract integer
  vector or `:` (type `Colon`).

**Warning.** `selectrows(model::SomeModelType, I, args...)` must always
return a tuple of the same length as `args`, even if this is one.

The fallback for `selectrows` is described at [`selectrows`](@ref).


### Example

Suppose a supervised model type `SomeSupervised` supports sample
weights, leading to two different `fit` signatures:

    fit(model::SomeSupervised, verbosity, X, y)
    fit(model::SomeSupervised, verbosity, X, y, w)

    predict(model::SomeSupervised, fitresult, Xnew)

Without a data front-end implemented, suppose `X` is expected to be a
table and `y` a vector, but suppose the core algorithm always converts
`X` to a matrix with features as rows (features corresponding to
columns in the table).  Then a new data-front end might look like
this:

    constant MMI = MLJModelInterface

    # for fit:
    MMI.reformat(::SomeSupervised, X, y) = (MMI.matrix(X, transpose=true), y)
    MMI.reformat(::SomeSupervised, X, y, w) = (MMI.matrix(X, transpose=true), y, w)
    MMI.selectrows(::SomeSupervised, I, Xmatrix, y) =
        (view(Xmatrix, :, I), view(y, I))
    MMI.selectrows(::SomeSupervised, I, Xmatrix, y, w) =
        (view(Xmatrix, :, I), view(y, I), view(w, I))

    # for predict:
    MMI.reformat(::SomeSupervised, X) = (MMI.matrix(X, transpose=true),)
    MMI.selectrows(::SomeSupervised, I, Xmatrix) = view(Xmatrix, I)

With these additions, `fit` and `predict` are refactored, so that `X`
and `Xnew` represent matrices with features as rows.

"""
reformat(model::Model, args...) = args

"""
    selectrows(::Model, I, data...) -> sampled_data

A model overloads `selectrows` whenever it buys into the optional
`reformat` front-end for data preprocessing. See [`reformat`](@ref)
for details. The fallback assumes `data` is a tuple and calls
`selectrows(X, I)` for each `X` in `data`, returning the results in a
new tuple of the same length. This call makes sense when `X` is a
table, abstract vector or abstract matrix. In the last two cases, a
new object and *not* a view is returned.

"""
selectrows(::Model, I, data...) = map(X -> selectrows(X, I), data)

# this operation can be optionally overloaded to provide access to
# fitted parameters (eg, coeficients of linear model):
"""
   fitted_params(model, fitresult) -> human_readable_fitresult # named_tuple

Models may overload `fitted_params`. The fallback returns
`(fitresult=fitresult,)`.

Other training-related outcomes should be returned in the `report`
part of the tuple returned by `fit`.

"""
fitted_params(::Model, fitresult) = (fitresult=fitresult,)

"""

    predict(model, fitresult, new_data...)

`Supervised` models must implement the `predict` operation. Here
`new_data` is the output of `reformat` called on user-specified data.

"""
function predict end

"""
probabilistic supervised models may overload `predict_mean`
"""
function predict_mean end

"""
probabilistic supervised models may overload `predict_mode`
"""
function predict_mode end

"""
probabilistic supervised models may overload `predict_median`
"""
function predict_median end

"""
`JointProbabilistic` supervised models MUST overload `predict_joint`.

`Probabilistic` supervised models MAY overload `predict_joint`.
"""
function predict_joint end

"""
unsupervised methods must implement the `transform` operation
"""
function transform end

"""
unsupervised methods may implement the `inverse_transform` operation
"""
function inverse_transform end

# models can optionally overload these for enable serialization in a
# custom format:
function save end
function restore end

"""
some meta-models may choose to implement the `evaluate` operations
"""
function evaluate end
