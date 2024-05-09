# The predict method

A compulsory `predict` method has the form

```julia
MMI.predict(model::SomeSupervisedModel, fitresult, Xnew) -> yhat
```

Here `Xnew` will have the same form as the `X` passed to `fit`.

Note that while `Xnew` generally consists of multiple observations
(e.g., has multiple rows in the case of a table) it is assumed, in view of
the i.i.d assumption recalled above, that calling `predict(..., Xnew)`
is equivalent to broadcasting some method `predict_one(..., x)` over
the individual observations `x` in `Xnew` (a method implementing the
probability distribution `p(X |y)` above).


## Prediction types for deterministic responses.

In the case of `Deterministic` models, `yhat` should have the same
scitype as the `y` passed to `fit` (see above). If `y` is a
`CategoricalVector` (classification) then elements of the prediction
`yhat` **must have a pool == to the pool of the target `y` presented
in training**, even if not all levels appear in the training data or
prediction itself.

Unfortunately, code not written with the preservation of categorical
levels in mind poses special problems. To help with this,
MLJModelInterface provides some utilities:
[`MLJModelInterface.int`](@ref) (for converting a `CategoricalValue`
into an integer, the ordering of these integers being consistent with
that of the pool) and `MLJModelInterface.decoder` (for constructing a
callable object that decodes the integers back into `CategoricalValue`
objects). Refer to [Convenience methods](@ref) below for important
details.

Note that a decoder created during `fit` may need to be bundled with
`fitresult` to make it available to `predict` during re-encoding. So,
for example, if the core algorithm being wrapped by `fit` expects a
nominal target `yint` of type `Vector{<:Integer}` then a `fit` method
may look something like this:

```julia
function MMI.fit(model::SomeSupervisedModel, verbosity, X, y)
    yint = MMI.int(y)
    a_target_element = y[1]                # a CategoricalValue/String
    decode = MMI.decoder(a_target_element) # can be called on integers

    core_fitresult = SomePackage.fit(X, yint, verbosity=verbosity)

    fitresult = (decode, core_fitresult)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end
```

while a corresponding deterministic `predict` operation might look like this:

```julia
function MMI.predict(model::SomeSupervisedModel, fitresult, Xnew)
    decode, core_fitresult = fitresult
    yhat = SomePackage.predict(core_fitresult, Xnew)
    return decode.(yhat)
end
```

For a concrete example, refer to the
[code](https://github.com/JuliaAI/MLJModels.jl/blob/master/src/ScikitLearn.jl)
for `SVMClassifier`.

Of course, if you are coding a learning algorithm from scratch, rather
than wrapping an existing one, these extra measures may be unnecessary.


## Prediction types for probabilistic responses

In the case of `Probabilistic` models with univariate targets, `yhat`
must be an `AbstractVector` or table whose elements are distributions.
In the common case of a vector (single target), this means one
distribution per row of `Xnew`.

A *distribution* is some object that, at the least, implements
`Base.rng` (i.e., is something that can be sampled).  Currently, all
performance measures (metrics) defined in MLJBase.jl additionally
assume that a distribution is either:

- An instance of some subtype of `Distributions.Distribution`, an
  abstract type defined in the
  [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/)
  package; or

- An instance of `CategoricalDistributions.UnivariateFinite`, from the
  [CategoricalDistributions.jl](https://github.com/JuliaAI/CategoricalDistributions.jl)
  package, *which should be used for all probabilistic classifiers*,
  i.e., for predictors whose target has scientific type
  `<:AbstractVector{<:Finite}`.

All such distributions implement the probability mass or density
function `Distributions.pdf`. If your model's predictions cannot be
predict objects of this form, then you will need to implement
appropriate performance measures to buy into MLJ's performance
evaluation apparatus.

An implementation can avoid CategoricalDistributions.jl as a
dependency by using the "dummy" constructor
`MLJModelInterface.UnivariateFinite`, which is bound to the true one
when MLJBase.jl is loaded.

For efficiency, one should not construct `UnivariateFinite` instances
one at a time. Rather, once a probability vector, matrix, or
dictionary is known, construct an instance of `UnivariateFiniteVector
<: AbstractArray{<:UnivariateFinite},1}` to return. Both
`UnivariateFinite` and `UnivariateFiniteVector` objects are
constructed using the single `UnivariateFinite` function.

For example, suppose the target `y` arrives as a subsample of some
`ybig` and is missing some classes:

```julia
ybig = categorical([:a, :b, :a, :a, :b, :a, :rare, :a, :b])
y = ybig[1:6]
```

Your fit method has bundled the first element of `y` with the
`fitresult` to make it available to `predict` for purposes of tracking
the complete pool of classes. Let's call this `an_element =
y[1]`. Then, supposing the corresponding probabilities of the observed
classes `[:a, :b]` are in an `n x 2` matrix `probs` (where `n` the number of
rows of `Xnew`) then you return

```julia
yhat = MLJModelInterface.UnivariateFinite([:a, :b], probs, pool=an_element)
```

This object automatically assigns zero-probability to the unseen class
`:rare` (i.e., `pdf.(yhat, :rare)` works and returns a zero
vector). If you would like to assign `:rare` non-zero probabilities,
simply add it to the first vector (the *support*) and supply a larger
`probs` matrix.

In a binary classification problem, it suffices to specify a single
vector of probabilities, provided you specify `augment=true`, as in
the following example, *and note carefully that these probabilities are
associated with the* **last** *(second) class you specify in the
constructor:*

```julia
y = categorical([:TRUE, :FALSE, :FALSE, :TRUE, :TRUE])
an_element = y[1]
probs = rand(10)
yhat = MLJModelInterface.UnivariateFinite([:FALSE, :TRUE], probs, augment=true, pool=an_element)
```

The constructor has a lot of options, including passing a dictionary
instead of vectors. See [`CategoricalDistributions.UnivariateFinite`](@ref)
for details.

See
[LinearBinaryClassifier](https://github.com/JuliaAI/MLJModels.jl/blob/master/src/GLM.jl)
for an example of a Probabilistic classifier implementation.

*Important note on binary classifiers.* There is no "Binary" scitype
distinct from `Multiclass{2}` or `OrderedFactor{2}`; `Binary` is just
an alias for `Union{Multiclass{2},OrderedFactor{2}}`. The
`target_scitype` of a binary classifier will generally be
`AbstractVector{<:Binary}` and according to the *mlj* scitype
convention, elements of `y` have type `CategoricalValue`, and *not*
`Bool`. See
[BinaryClassifier](https://github.com/JuliaAI/MLJModels.jl/blob/master/src/GLM.jl)
for an example.

## Report items returned by predict

A `predict` method, or other operation such as `transform`, can contribute to the report
accessible in any machine associated with a model. See [Reporting byproducts of a
static transformation](@ref) below for details.


