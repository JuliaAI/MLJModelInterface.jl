# Models that learn a probability distribution


!!! warning "Experimental"

	The following API is experimental. It is subject to breaking changes during minor or major releases without warning. Models implementing this interface will not work with MLJBase versions earlier than 0.17.5.

Models that fit a probability distribution to some `data` should be
regarded as `Probabilistic <: Supervised` models with target `y = data`
and `X = nothing`.

The `predict` method should return a single distribution.

A working implementation of a model that fits a `UnivariateFinite` distribution to some
categorical data using [Laplace
smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) controlled by a
hyperparameter `alpha` is given in [MLJBase
tests](https://github.com/JuliaAI/MLJBase.jl/blob/dev/test/resampling.jl); try
[here](https://github.com/JuliaAI/MLJBase.jl/blob/203aae371f67ed639685aa4803e150ff69f0fa49/test/resampling.jl#L1050).


