# every model interface must implement a `fit` method of the form
# `fit(model, verb::Integer, training_args...) -> fitresult, cache, report`
function fit end

# fallback for static transformations
fit(::Static, ::Integer, a...) = (nothing, nothing, nothing)

# fallbacks for supervised models that don't support sample weights:
fit(m::Supervised, verb::Integer, X, y, w) = fit(m, verb, X, y)

# this operation can be optionally overloaded to provide access to
# fitted parameters (eg, coeficients of linear model):
fitted_params(::Model, fitres) = (fitresult=fitres,)

# each model interface may overload the following refitting method:
update(m::Model, verb::Integer, fitres, cache, a...) = fit(m, verb, a...)

# stub for online learning method update method
function update_data end

# supervised methods must implement the predict operation; additionally,
# probabilistic supervised models may overload one or more of
# `predict_mode`, `predict_median` and `predict_mean`
function predict end
function predict_mean end
function predict_mode end
function predict_median end

# unsupervised methods must implement the transform operation and
# may implement the inverse_transform operation
function transform end
function inverse_transform end

# models can optionally overload these for enable serialization in a
# custom format:
function save end
function restore end

# operations implemented by some meta-models:
function evaluate end
