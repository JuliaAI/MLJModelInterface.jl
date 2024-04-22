# The fit method

A compulsory `fit` method returns three objects:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y) -> fitresult, cache, report
```

1. `fitresult` is the fitresult in the sense above (which becomes an
    argument for `predict` discussed below).

2.  `report` is a (possibly empty) `NamedTuple`, for example,
    `report=(deviance=..., dof_residual=..., stderror=..., vcov=...)`.
    Any training-related statistics, such as internal estimates of the
    generalization error, and feature rankings, should be returned in
    the `report` tuple. How, or if, these are generated should be
    controlled by hyperparameters (the fields of `model`). Fitted
    parameters, such as the coefficients of a linear model, do not go
    in the report as they will be extractable from `fitresult` (and
    accessible to MLJ through the `fitted_params` method described below).

3.  The value of `cache` can be `nothing`, unless one is also defining
    an `update` method (see below). The Julia type of `cache` is not
    presently restricted.

!!! note

	The  `fit` (and `update`) methods should not mutate the `model`. If necessary, `fit` can create a `deepcopy` of `model` first.


It is not necessary for `fit` to provide type or dimension checks on
`X` or `y` or to call `clean!` on the model; MLJ will carry out such
checks.

The types of `X` and `y` are constrained by the `input_scitype` and
`target_scitype` trait declarations; see [Trait declarations](@ref)
below. (That is, unless a data front-end is implemented, in which case
these traits refer instead to the arguments of the overloaded
`reformat` method, and the types of `X` and `y` are determined by the
output of `reformat`.)

The method `fit` should never alter hyperparameter values, the sole
exception being fields of type `<:AbstractRNG`. If the package is able
to suggest better hyperparameters, as a byproduct of training, return
these in the report field.

The `verbosity` level (0 for silent) is for passing to the learning
algorithm itself. A `fit` method wrapping such an algorithm should
generally avoid doing any of its own logging.

*Sample weight support.* If
`supports_weights(::Type{<:SomeSupervisedModel})` has been declared
`true`, then one instead implements the following variation on the
above `fit`:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y, w=nothing) -> fitresult, cache, report
```
