# New model type declarations

Here is an example of a concrete supervised model type declaration,
for a model with a single hyperparameter:

```julia
import MLJModelInterface
const MMI = MLJModelInterface

mutable struct RidgeRegressor <: MMI.Deterministic
    lambda::Float64
end
```

Models (which are mutable) should not be given internal constructors.
It is recommended that they be given an external lazy keyword constructor
of the same name. This constructor defines default values for every field,
and optionally corrects invalid field values by calling a `clean!`
method (whose fallback returns an empty message string):

```julia
function MMI.clean!(model::RidgeRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

# keyword constructor
function RidgeRegressor(; lambda=0.0)
    model = RidgeRegressor(lambda)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end
```

*Important.* Performing `clean!(model)` a second time should not mutate `model`. That is,
this test should hold:

```julia
clean!(model)
clone = deepcopy(model)
clean!(model)
@test model == clone
```

Although not essential, try to avoid `Union` types for model
fields. For example, a field declaration `features::Vector{Symbol}`
with a default of `Symbol[]` (detected with `isempty` method) is
preferred to `features::Union{Vector{Symbol}, Nothing}` with a default
of `nothing`.

### Hyperparameters for parallelization options

The section [Acceleration and
Parallelism](https://JuliaAI.github.io/MLJ.jl/dev/acceleration_and_parallelism/)
of the MLJ manual indicates how users specify an option to run an algorithm using
distributed processing or multithreading. A hyperparameter specifying such an option
should be called `acceleration`. Its value `a` should satisfy `a isa AbstractResource`
where `AbstractResource` is defined in the ComputationalResources.jl package. An option to
run on a GPU is ordinarily indicated with the `CUDALibs()` resource.

### hyperparameter access and mutation

To support hyperparameter optimization (see the [Tuning
Models](https://JuliaAI.github.io/MLJ.jl/dev/tuning_models/) section of the
MLJ manual) any hyperparameter to be individually controlled must be:

- property-accessible; nested property access allowed, as in
  `model.detector.K`

- mutable

For an un-nested hyperparameter, the requirement is that
`getproperty(model, :param_name)` and `setproperty!(model,
:param_name, value)` have the expected behavior.

Combining hyperparameters in a named tuple does not generally
work: although property-accessible (with nesting), an
individual value cannot be mutated.

For a suggested way to deal with hyperparameters varying in number, see the
[implementation](https://github.com/JuliaAI/MLJBase.jl/blob/dev/src/composition/models/stacking.jl)
of `Stack`, where the model struct stores a varying number of base models internally as a
vector, but components are named at construction and accessed by overloading
`getproperty/setproperty!`  appropriately.

### Macro shortcut

An alternative to declaring the model struct, clean! method and
keyword constructor, is to use the `@mlj_model` macro, as in the
following example:

```julia
@mlj_model mutable struct YourModel <: MMI.Deterministic
    a::Float64 = 0.5::(_ > 0)
    b::String  = "svd"::(_ in ("svd","qr"))
end
```

This declaration specifies:

* A keyword constructor (here `YourModel(; a=..., b=...)`),
* Default values for the hyperparameters,
* Constraints on the hyperparameters where `_` refers to a value
  passed.

For example, `a::Float64 = 0.5::(_ > 0)` indicates that
the field `a` is a `Float64`, takes `0.5` as default value, and
expects its value to be positive.

You cannot use the `@mlj_model` macro if your model struct has type
parameters.

#### Known issue with `@mlj_macro`

Defaults with negative values can trip up the `@mlj_macro` (see [this
issue](https://github.com/JuliaAI/MLJBase.jl/issues/68)). So,
for example, this does not work:

```julia
@mlj_model mutable struct Bar
    a::Int = -1::(_ > -2)
end
```

But this does:

```julia
@mlj_model mutable struct Bar
    a::Int = (-)(1)::(_ > -2)
end
```
