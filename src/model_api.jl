"""
every model interface must implement a `fit` method of the form
`fit(model, verb::Integer, training_args...) -> fitresult, cache, report`
"""
function fit end

# fallback for static transformations
fit(::Static, ::Integer, a...) = (nothing, nothing, nothing)

# fallbacks for supervised models that don't support sample weights:
fit(m::Supervised, verb::Integer, X, y, w) = fit(m, verb, X, y)

# this operation can be optionally overloaded to provide access to
# fitted parameters (eg, coeficients of linear model):
fitted_params(::Model, fitres) = (fitresult=fitres,)

"""
each model interface may overload the `update` refitting method
"""
update(m::Model, verb::Integer, fitres, cache, a...) = fit(m, verb, a...)

"""
each model interface may overload the `update_data` refitting method for online learning
"""
function update_data end

"""
supervised methods must implement the `predict` operation
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
