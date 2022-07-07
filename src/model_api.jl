"""
    MLJModelInterface.fit(model, verbosity, data...) -> fitresult, cache, report

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

# fallback for unsupervised annotators when labels or weights appear:
# this is useful for evaluation and mixed composite models that combine
# both supervised and unsupervised annotators
fit(m::UnsupervisedAnnotator, verbosity, X, y) = fit(m, verbosity, X)
fit(m::UnsupervisedAnnotator, verbosity, X, y, w) = fit(m, verbosity, X)

"""
    MLJModelInterface.update(model, verbosity, fitresult, cache, data...)

Models may optionally implement an `update` method. The fallback calls
`fit`.

"""
function update(m::Model, verbosity, fitresult, cache, data...)
    return fit(m, verbosity, data...)
end

"""
    MLJModelInterface.training_losses(model::M, report)

If `M` is an iterative model type which calculates training losses,
implement this method to return an `AbstractVector` of the losses
in historical order. If the model calculates scores instead, then the
sign of the scores should be reversed.

The following trait overload is also required:
`supports_training_losses(::Type{<:M}) = true`

"""
training_losses(model, report) = nothing

# to support online learning in the future:
# https://github.com/alan-turing-institute/MLJ.jl/issues/60 :
function update_data end

"""
    MLJModelInterface.reformat(model, args...) -> data

Models optionally overload `reformat` to define transformations of
user-supplied data into some model-specific representation (e.g., from
a table to a matrix). When implemented, the MLJ user can avoid
repeating such transformations unnecessarily, and can additionally
make use of more efficient row subsampling, which is then based on the
model-specific representation of data, rather than the
user-representation. When `reformat` is overloaded,
`selectrows(::Model, ...)` must be as well (see
[`selectrows`](@ref)). Furthermore, the model `fit` method(s), and
operations, such as `predict` and `transform`, must be refactored to
act on the model-specific representions of the data.

To implement the `reformat` data front-end for a model, refer to
"Implementing a data front-end" in the [MLJ
manual](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/).


"""
reformat(model::Model, args...) = args

"""
    MLJModelInterface.selectrows(::Model, I, data...) -> sampled_data

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

`Supervised` and `SupervisedAnnotator` models must implement the
`predict` operation. Here `new_data` is the output of `reformat`
called on user-specified data.

"""
function predict end

"""

Models types `M` for which `prediction_type(M) == :probablisitic` may
overload `predict_mean`.

"""
function predict_mean end

"""

Models types `M` for which `prediction_type(M) == :probablisitic` may
overload `predict_mode`.

"""
function predict_mode end

"""

Models types `M` for which `prediction_type(M) == :probablisitic` may
overload `predict_median`.

"""
function predict_median end

"""
`JointProbabilistic` supervised models MUST overload `predict_joint`.

`Probabilistic` supervised models MAY overload `predict_joint`.
"""
function predict_joint end

"""
`Unsupervised` models must implement the `transform` operation.
"""
function transform end

"""

`Unsupervised` models may implement the `inverse_transform` operation.

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

"""
    feature_importances(model::M, fitresult, report)

For a given `model` of model type `M` supporting intrinsic feature importances, calculate 
the feature importances from the model's `fitresult` and `report` as an 
abstract vector of `feature::Symbol => importance::Real` pairs 
(e.g `[:gender =>0.23, :height =>0.7, :weight => 0.1]`).  

The following trait overload is also required:
`reports_feature_importances(::Type{<:M}) = true`

"""
feature_importances(model, fitresult, report) = nothing
