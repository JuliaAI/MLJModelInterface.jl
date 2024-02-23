# Outlier detection models

!!! warning "Experimental API"

	The Outlier Detection API is experimental and may change in future
	releases of MLJ.

Outlier detection or *anomaly detection* is predominantly an unsupervised
learning task, transforming each data point to an outlier score quantifying
the level of "outlierness". However, because detectors can also be
semi-supervised or supervised, MLJModelInterface provides a collection of
abstract model types, that enable the different characteristics, namely:

- `MLJModelInterface.SupervisedDetector`
- `MLJModelInterface.UnsupervisedDetector`
- `MLJModelInterface.ProbabilisticSupervisedDetector`
- `MLJModelInterface.ProbabilisticUnsupervisedDetector`
- `MLJModelInterface.DeterministicSupervisedDetector`
- `MLJModelInterface.DeterministicUnsupervisedDetector`

All outlier detection models subtyping from any of the above supertypes
have to implement `MLJModelInterface.fit(model, verbosity, X, [y])`.
Models subtyping from either `SupervisedDetector` or `UnsupervisedDetector`
have to implement `MLJModelInterface.transform(model, fitresult, Xnew)`, which
should return the raw outlier scores (`<:Continuous`) of all points in `Xnew`.

Probabilistic and deterministic outlier detection models provide an additional
option to predict a normalized estimate of outlierness or a concrete
outlier label and thus enable evaluation of those models. All corresponding
supertypes have to implement (in addition to the previously described `fit`
and `transform`) `MLJModelInterface.predict(model, fitresult, Xnew)`, with
deterministic predictions conforming to `OrderedFactor{2}`, with the first
class being the normal class and the second class being the outlier.
Probabilistic models predict a `UnivariateFinite` estimate of those classes.

It is typically possible to automatically convert an outlier detection model
to a probabilistic or deterministic model if the training scores are stored in
the model's `report`. Below mentioned `OutlierDetection.jl` package, for example,
stores the training scores under the `scores` key in the `report` returned from
`fit`. It is then possible to use model wrappers such as
`OutlierDetection.ProbabilisticDetector` to automatically convert a model to
enable predictions of the required output type.

!!! note "External outlier detection packages"

	[OutlierDetection.jl](https://github.com/OutlierDetectionJL/OutlierDetection.jl)
	provides an opinionated interface on top of MLJ for outlier detection models,
	standardizing things like class names, dealing with training scores, score
	normalization and more.
