# Model wrappers

A model that can have one or more other models as hyper-parameters should overload the trait `is_wrapper`, as in this example:

```julia
MLJModelInterface.target_in_fit(::Type{<:MyWrapper}) = true
```

The constructor for such a model does not need provide default values for the model-valued
hyper-parameters. If only a single model is wrapped, then the hyper-parameter should have
the name `:model` and this should be an optional positional argument, as well as a keyword
argument.

For example, `EnsembleModel` is a model wrapper, and we can construct an instance like this:

```julia
using MLJ
atom = ConstantClassfier()
EnsembleModel(tree, n=100)
```

but also like this:

```julia
EnsembleModel(model=tree, n=100)
```

This is the only case in MLJ where positional arguments in a model constructor are
allowed.

## Handling generic constructors

Model wrappers frequently have a public facing constructor with a name different from that
of the model type constructed. For example, `TunedModel(model, ...)` is a constructor that
will construct either an instance of `DeterministicTunedModel` or
`ProbabilisticTunedModel`, depending on the type of `model`. In this case it is necessary
to overload the `constructor` trait, which in that case looks like this:

```julia
MLJModelInterface.constructor(::Type{<:Union{
    DeterministicTunedModel,
	ProbabilisticTunedModel,
	}}) = TunedModel
```

This allows the MLJ Model Registry to correctly associate model metadata to the
constructor, rather than the (private) types.
