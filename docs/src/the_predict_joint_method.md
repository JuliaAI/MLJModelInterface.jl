# The predict_joint method

!!! warning "Experimental"

	The following API is experimental. It is subject to breaking changes during minor or major releases without warning.

```julia
MMI.predict_joint(model::SomeSupervisedModel, fitresult, Xnew) -> yhat
```

Any `Probabilistic` model type `SomeModel`may optionally implement a
`predict_joint` method, which has the same signature as `predict`, but
whose predictions are a single distribution (rather than a vector of
per-observation distributions).

Specifically, the output `yhat` of `predict_joint` should be an
instance of `Distributions.Sampleable{<:Multivariate,V}`, where
`scitype(V) = target_scitype(SomeModel)` and samples have length `n`,
where `n` is the number of observations in `Xnew`.

If a new model type subtypes `JointProbabilistic <: Probabilistic` then
implementation of `predict_joint` is compulsory.


