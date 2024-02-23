# Static models

A model type subtypes `Static <: Unsupervised` if it does not generalize to new data but
nevertheless has hyperparameters. See the [Static
transformers](https://alan-turing-institute.github.io/MLJ.jl/dev/transformers/#Static-transformers)
section of the MLJ manual for examples. In the `Static` case, `transform` can have
multiple arguments and `input_scitype` refers to the allowed scitype of the slurped data,
*even if there is only a single argument.* For example, if the signature is
`transform(static_model, X1, X2)`, then the allowed `input_scitype` might be
`Tuple{Table(Continuous), Table(Continuous)}`; if the signature is
`transform(static_model, X)`, the allowed `input_scitype` might be
`Tuple{Table(Continuous)}`. The other traits are as for regular [Unsupervised
models](@ref).

## Reporting byproducts of a static transformation

As a static transformer does not implement `fit`, the usual mechanism for creating a
`report` is not available. Instead, byproducts of the computation performed by `transform`
can be returned by `transform` itself by returning a pair (`output`, `report`) instead of
just `output`.  Here `report` should be a named tuple. In fact, any operation, (e.g.,
`predict`) can do this for any model type. However, this exceptional behavior must be
flagged with an appropriate trait declaration, as in

```julia
MLJModelInterface.reporting_operations(::Type{<:SomeModelType}) = (:transform,)
```

If `mach` is a machine wrapping a model of this kind, then the `report(mach)` will include
the `report` item form `transform`'s output. For sample implementations, see [this
issue](https://github.com/JuliaAI/MLJBase.jl/pull/806) or the code for [DBSCAN
clustering](https://github.com/jbrea/MLJClusteringInterface.jl/blob/41d3c2195ad33f1840596c9762a3a67b9a124c6a/src/MLJClusteringInterface.jl#L125).
