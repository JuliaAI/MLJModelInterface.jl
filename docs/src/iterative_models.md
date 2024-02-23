# Iterative models and the update! method

An `update` method may be optionally overloaded to enable a call by
MLJ to retrain a model (on the same training data) to avoid repeating
computations unnecessarily.

```julia
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y) -> fit
result, cache, report
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y, w=nothing) -> fit
result, cache, report
```

Here the second variation applies if `SomeSupervisedModel` supports
sample weights.

If an MLJ `Machine` is being `fit!` and it is not the first time, then `update` is called
instead of `fit`, unless the machine `fit!` has been called with a new `rows` keyword
argument. However, `MLJModelInterface` defines a fallback for `update` which just calls
`fit`. For context, see the
[Internals](https://alan-turing-institute.github.io/MLJ.jl/dev/internals/) section of the
MLJ manual.

Learning networks wrapped as models constitute one use case (see the [Composing
Models](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/) section of
the MLJ manual): one would like each component model to be retrained only when
hyperparameter changes "upstream" make this necessary. In this case, MLJ provides a
fallback (specifically, the fallback is for any subtype of `SupervisedNetwork =
Union{DeterministicNetwork,ProbabilisticNetwork}`). A second more generally relevant use
case is iterative models, where calls to increase the number of iterations only restarts
the iterative procedure if other hyperparameters have also changed. (A useful method for
inspecting model changes in such cases is `MLJModelInterface.is_same_except`. ) For an
example, see [MLJEnsembles.jl](https://github.com/JuliaAI/MLJEnsembles.jl).

A third use case is to avoid repeating the time-consuming preprocessing of
`X` and `y` required by some models.

If the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required (for example, pre-processed versions
of `X` and `y`), as this is also passed as an argument to the `update`
method.

