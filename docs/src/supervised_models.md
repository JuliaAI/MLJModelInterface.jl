# Supervised models

## Mathematical assumptions

At present, MLJ's performance estimate functionality (resampling using
`evaluate`/`evaluate!`) tacitly assumes that feature-label pairs of observations `(X1,
y1), (X2, y2), (X2, y2), ...` are being modelled as identically independent random
variables (i.i.d.), and constructs some kind of representation of an estimate of the
conditional probability `p(y | X)` (`y` and `X` *single* observations). It may be that a
model implementing the MLJ interface has the potential to make predictions under weaker
assumptions (e.g., time series forecasting models). However the output of the compulsory
`predict` method described below should be the output of the model under the i.i.d
assumption.

In the future, newer methods may be introduced to handle weaker assumptions (see, e.g.,
[The predict_joint method](@ref) below).

The following sections were written with `Supervised` models in mind, but also cover
material relevant to general models:

- [Summary of methods](@ref)
- [The form of data for fitting and predicting](@ref) 
- [The fit method](@ref)
- [The fitted_params method](@ref)
- [The predict method](@ref) 
- [The predict_joint method](@ref) 
- [Training losses](@ref) 
- [Feature importances](@ref) 
- [Trait declarations](@ref) 
- [Iterative models and the update! method](@ref) 
- [Implementing a data front end](@ref) 
- [Supervised models with a transform method](@ref) 
- [Models that learn a probability distribution](@ref)
