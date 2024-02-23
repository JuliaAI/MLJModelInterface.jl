# Models that learn a probability distribution


!!! warning "Experimental"

	The following API is experimental. It is subject to breaking changes during minor or major releases without warning. Models implementing this interface will not work with MLJBase versions earlier than 0.17.5.

Models that fit a probability distribution to some `data` should be
regarded as `Probabilistic <: Supervised` models with target `y = data`
and `X = nothing`.

The `predict` method should return a single distribution.

A working implementation of a model that fits a `UnivariateFinite`
distribution to some categorical data using [Laplace
smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
controlled by a hyperparameter `alpha` is given
[here](https://github.com/JuliaAI/MLJBase.jl/blob/d377bee1198ec179a4ade191c11fef583854af4a/test/interface/model_api.jl#L36).


