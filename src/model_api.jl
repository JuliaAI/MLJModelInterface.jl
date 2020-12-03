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
user-supplied training data `args` into some model-specific
representation `data` (e.g., from a table to a matrix). The fallback
returns `args` (no transformation).

If `mach` is a machine with `mach.model == model` then calling `fit!(mach)`
will either call

    fit(model, verbosity, data...)

or

    update(model, verbosity, data...)

where `data = reformat(model, mach.args...)`. This means that
overloading `reformat` alters the form of the arguments expected by
`fit` and `update`.

If, instead, one calls `fit!(mach, rows=I)`, then `data` in the above
`fit`/`update` calls is replaced with `selectrows(model, I,
data...)`. So overloading `reformat` generally requires overloading of
`selectrows` also, to specify how the model-specific representation of
training data is resampled.

Note `data` is always a tuple, but it needn't have the same length as
`args`.

"""
reformat(model::Model, args...) = args

"""
    selectrows(::Model, I, data...) -> sampled_data

Models optionally overload `selectrows` for efficient resampling of
training data. Here `data` is the ouput of calling `reformat` on
user-provided data.  The fallback assumes `data` is a tuple and calls
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
