# Supervised models with a transform method

A supervised model may optionally implement a `transform` method,
whose signature is the same as `predict`. In that case, the
implementation should define a value for the `output_scitype` trait. A
declaration

```julia
output_scitype(::Type{<:SomeSupervisedModel}) = T
```

is an assurance that `scitype(transform(model, fitresult, Xnew)) <: T`
always holds, for any `model` of type `SomeSupervisedModel`.

A use-case for a `transform` method for a supervised model is a neural
network that learns *feature embeddings* for categorical input
features as part of overall training. Such a model becomes a
transformer that other supervised models can use to transform the
categorical features (instead of applying the higher-dimensional one-hot
encoding representations).

