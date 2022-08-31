"""
    metadata_pkg(T; args...)

Helper function to write the metadata for a package providing model `T`.
Use it with broadcasting to define the metadata of the package providing
a series of models.

## Keywords

* `package_name="unknown"`   : package name
* `package_uuid="unknown"`   : package uuid
* `package_url="unknown"`    : package url
* `is_pure_julia=missing`    : whether the package is pure julia
* `package_license="unknown"`: package license
* `is_wrapper=false` : whether the package is a wrapper

## Example

```julia
metadata_pkg.((KNNRegressor, KNNClassifier),
    package_name="NearestNeighbors",
    package_uuid="b8a86587-4115-5ab1-83bc-aa920d37bbce",
    package_url="https://github.com/KristofferC/NearestNeighbors.jl",
    is_pure_julia=true,
    package_license="MIT",
    is_wrapper=false)
```
"""
function metadata_pkg(
    T;
    # aliases:
    name::String="unknown",
    uuid::String="unknown",
    url::String="unknown",
    julia::Union{Missing,Bool}=missing,
    license::String="unknown",
    is_wrapper::Bool=false,

    # preferred names, corresponding to trait names:
    package_name=name,
    package_uuid=uuid,
    package_url=url,
    is_pure_julia=julia,
    package_license=license,
)
    ex = quote
        $MLJModelInterface.package_name(::Type{<:$T}) = $package_name
        $MLJModelInterface.package_uuid(::Type{<:$T}) = $package_uuid
        $MLJModelInterface.package_url(::Type{<:$T}) = $package_url
        $MLJModelInterface.is_pure_julia(::Type{<:$T}) = $is_pure_julia
        $MLJModelInterface.package_license(::Type{<:$T}) = $package_license
        $MLJModelInterface.is_wrapper(::Type{<:$T}) = $is_wrapper
    end
    parentmodule(T).eval(ex)
end

# Extend `program` (an expression) to include trait definition for
# specified `trait` and type `T`.
function _extend!(program::Expr, trait::Symbol, value, T)
    if value !== nothing
        push!(program.args, quote
              $MLJModelInterface.$trait(::Type{<:$T}) = $value
              end)
        return nothing
    end
end

const depwarn_docstring(T) =
    """

    Regarding $T: `metadata_model` should not be called with the keyword argument `descr`
    or `docstring`. Implementers of the MLJ model interface should instead create an
    MLJ-compliant docstring in the usual way.  See
    https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Document-strings
    for details.

    """
    metadata_model(T; args...)

Helper function to write the metadata for a model `T`.

## Keywords

* `input_scitype=Unknown`: allowed scientific type of the input data
* `target_scitype=Unknown`: allowed scitype of the target (supervised)
* `output_scitype=Unkonwn`: allowed scitype of the transformed data (unsupervised)
* `supports_weights=false`: whether the model supports sample weights
* `supports_class_weights=false`: whether the model supports class weights
* `load_path="unknown"`: where the model is (usually `PackageName.ModelName`)
* `human_name=nothing`: human name of the model
* `supports_training_losses=nothing`: whether the (necessarily iterative) model can report
  training losses
* `reports_feature_importances=nothing`: whether the model reports feature importances

## Example

```julia
metadata_model(KNNRegressor,
    input_scitype=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target_scitype=AbstractVector{MLJModelInterface.Continuous},
    supports_weights=true,
    load_path="NearestNeighbors.KNNRegressor")
```
"""
function metadata_model(
    T;
    # aliases:
    input=nothing,
    target=nothing,
    output=nothing,
    weights::Union{Nothing,Bool}=nothing,
    class_weights::Union{Nothing,Bool}=nothing,
    descr::Union{Nothing,String}=nothing,
    path::Union{Nothing,String}=nothing,

    # preferred names, corresponding to trait names:
    input_scitype=input,
    target_scitype=target,
    output_scitype=output,
    supports_weights::Union{Nothing,Bool}=weights,
    supports_class_weights::Union{Nothing,Bool}=class_weights,
    docstring::Union{Nothing,String}=descr,
    load_path::Union{Nothing,String}=path,
    human_name::Union{Nothing,String}=nothing,
    supports_training_losses::Union{Nothing,Bool}=nothing,
    reports_feature_importances::Union{Nothing,Bool}=nothing,
)
    docstring === nothing || Base.depwarn(depwarn_docstring(T), :metadata_model)

    program = quote end

    # Note: Naively using metaprogramming to roll up the following
    # code does not work. Only change this if you really know what
    # you're doing.
    _extend!(program, :input_scitype, input_scitype, T)
    _extend!(program, :target_scitype, target_scitype, T)
    _extend!(program, :output_scitype, output_scitype, T)
    _extend!(program, :supports_weights, supports_weights, T)
    _extend!(program, :supports_class_weights,supports_class_weights, T)
    _extend!(program, :docstring, docstring, T)
    _extend!(program, :load_path, load_path, T)
    _extend!(program, :human_name, human_name, T)
    _extend!(program, :supports_training_losses, supports_training_losses, T)
    _extend!(program, :reports_feature_importances, reports_feature_importances, T)

    parentmodule(T).eval(program)
end

# TODO: After `human_name` trait is added as model trait, include in
# example given in the docstring for `doc_header`.

"""
    MLJModelInterface.doc_header(SomeModelType)

Return a string suitable for interpolation in the document string of
an MLJ model type. In the example given below, the header expands to
something like this:

>    `FooRegressor`
>
>A model type for constructing a foo regressor,
>based on [FooRegressorPkg.jl](http://existentialcomics.com/).
>
>From MLJ, the type can be imported using
>
>
>    `FooRegressor = @load FooRegressor pkg=FooRegressorPkg`
>
>Construct an instance with default hyper-parameters using the syntax
>`model = FooRegressor()`. Provide keyword arguments to override
>hyper-parameter defaults, as in `FooRegressor(a=...)`.

Ordinarily, `doc_header` is used in document strings defined *after*
the model type definition, as `doc_header` assumes model traits (in
particular, `package_name` and `package_url`) to be defined; see also
[`MLJModelInterface.metadata_pkg`](@ref).


### Example

Suppose a model type and traits have been defined by:

```
mutable struct FooRegressor
    a::Int
    b::Float64
end

metadata_pkg(FooRegressor,
    name="FooRegressorPkg",
    uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2",
    url="http://existentialcomics.com/",
    )
metadata_model(FooRegressor,
    input=Table(Continuous),
    target=AbstractVector{Continuous},
    descr="La di da")
```

Then the docstring is defined post-facto with the following code:

```
\"\"\"
\$(MLJModelInterface.doc_header(FooRegressor))

### Training data

In MLJ or MLJBase, bind an instance `model` ...

<rest of doc string goes here>

\"\"\"
FooRegressor

```

"""
function doc_header(SomeModelType)
    name = MLJModelInterface.name(SomeModelType)
    human_name = MLJModelInterface.human_name(SomeModelType)
    package_name = MLJModelInterface.package_name(SomeModelType)
    package_url = MLJModelInterface.package_url(SomeModelType)
    params = MLJModelInterface.hyperparameters(SomeModelType)

    ret =
        """
        ```
        $name
        ```

        A model type for constructing a $human_name, based on
        [$(package_name).jl]($package_url), and implementing the MLJ
        model interface.

        From MLJ, the type can be imported using

        ```
        $name = @load $name pkg=$package_name
        ```

        Do `model = $name()` to construct an instance with default hyper-parameters.
        """ |> chomp

    ret *= " "

    isempty(params) && return ret

    p = first(params)
    ret *=
        """
        Provide keyword arguments to override hyper-parameter defaults, as in
        `$name($p=...)`.
        """ |> chomp

    return ret
end

"""
    synthesize_docstring

Private method.

Generates a value for the `docstring` trait for use with a model which
does not have a standard document string, to use as the fallback. See
[`metadata_model`](@ref).

"""
function synthesize_docstring(M)
    package_name = MLJModelInterface.package_name(M)
    package_url  = MLJModelInterface.package_url(M)
    model_name   = MLJModelInterface.name(M)
    human_name   = MLJModelInterface.human_name(M)
    hyperparameters = MLJModelInterface.hyperparameters(M)

    text_for_params = ""
    model = try
        M()
    catch ex
        return ""
    end

    # generate text for the section on hyperparameters
    if !is_wrapper(M)
        isempty(hyperparameters) || (text_for_params *= "# Hyper-parameters")
        for p in hyperparameters
            value = getproperty(model, p)
            text_for_params *= "\n\n- `$p = $value`"
        end
    end

    ret = doc_header(M)
    if !isempty(text_for_params)
        ret *=
            """

            $text_for_params

            """
    end
    return ret
end
