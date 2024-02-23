# The fitted_params method

A `fitted_params` method may be optionally overloaded. Its purpose is
to provide MLJ access to a user-friendly representation of the
learned parameters of the model (as opposed to the
hyperparameters). They must be extractable from `fitresult`.

```julia
MMI.fitted_params(model::SomeSupervisedModel, fitresult) -> friendly_fitresult::NamedTuple
```

For a linear model, for example, one might declare something like
`friendly_fitresult=(coefs=[...], bias=...)`.

The fallback is to return `(fitresult=fitresult,)`.
