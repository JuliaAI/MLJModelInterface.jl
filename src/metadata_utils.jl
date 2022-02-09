"""
    docstring_ext

Internal function to help generate the docstring for a package. See
[`metadata_model`](@ref).
"""
function docstring_ext(T; descr::String="")
    package_name = MLJModelInterface.package_name(T)
    package_url  = MLJModelInterface.package_url(T)
    model_name   = MLJModelInterface.name(T)
    # the message to return
    message = "$descr"
    message *= "\n→ based on [$package_name]($package_url)."
    message *= "\n→ do `@load $model_name pkg=\"$package_name\"` to " *
        "use the model."
    message *= "\n→ do `?$model_name` for documentation."
end

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
        MLJModelInterface.package_name(::Type{<:$T}) = $package_name
        MLJModelInterface.package_uuid(::Type{<:$T}) = $package_uuid
        MLJModelInterface.package_url(::Type{<:$T}) = $package_url
        MLJModelInterface.is_pure_julia(::Type{<:$T}) = $is_pure_julia
        MLJModelInterface.package_license(::Type{<:$T}) = $package_license
        MLJModelInterface.is_wrapper(::Type{<:$T}) = $is_wrapper
    end
    parentmodule(T).eval(ex)
end

"""
    metadata_model(`T`; args...)

Helper function to write the metadata for a model `T`.

## Keywords

* `input_scitype=Unknown` : allowed scientific type of the input data
* `target_scitype=Unknown`: allowed sc. type of the target (supervised)
* `output_scitype=Unknown`: allowed sc. type of the transformed data (unsupervised)
* `supports_weights=false` : whether the model supports sample weights
* `docstring=""` : short description of the model
* `load_path=""` : where the model is (usually `PackageName.ModelName`)

## Example

```julia
metadata_model(KNNRegressor,
    input_scitype=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target_scitype=AbstractVector{MLJModelInterface.Continuous},
    supports_weights=true,
    docstring="K-Nearest Neighbors classifier: ...",
    load_path="NearestNeighbors.KNNRegressor")
```
"""
function metadata_model(
    T;
    # aliases:
    input=Unknown,
    target=Unknown,
    output=Unknown,
    weights::Bool=false,
    descr::String="",
    path::String="",

    # preferred names, corresponding to trait names:
    input_scitype=input,
    target_scitype=target,
    output_scitype=output,
    supports_weights=weights,
    docstring=descr,
    load_path=path,
)
    if isempty(load_path)
        pname = MLJModelInterface.package_name(T)
        mname = MLJModelInterface.name(T)
        load_path = "MLJModels.$(pname)_.$(mname)"
    end
    ex = quote
        MLJModelInterface.input_scitype(::Type{<:$T}) = $input_scitype
        MLJModelInterface.output_scitype(::Type{<:$T}) = $output_scitype
        MLJModelInterface.target_scitype(::Type{<:$T}) = $target_scitype
        MLJModelInterface.supports_weights(::Type{<:$T}) = $supports_weights
        MLJModelInterface.load_path(::Type{<:$T}) = $load_path

        function MLJModelInterface.docstring(::Type{<:$T})
            return MLJModelInterface.docstring_ext($T; descr=$docstring)
        end
    end
    parentmodule(T).eval(ex)
end

function doc_header(model)
    name = MLJModelInterface.name(model)
    human_name = MLJModelInterface.human_name(model)
    package_name = MLJModelInterface.package_name(model)
    package_url = MLJModelInterface.package_url(model)
    params = MLJModelInterface.hyperparameters(model)

    ret =
"""
    $name

Model type for $human_name, based on [$package_name]($package_url).

From MLJ, the type can be imported using

    $name = @load $name pkg=$package_name

Construct an instance with default hyper-parameters using the syntax
`model = $name()`.
""" |> chomp

    isempty(params) && return ret

    p = first(params)
    ret *=
"""
 Provide keyword arguments to override hyper-parameter defaults, as in
`$name($p=...)`.
""" |> chomp

    return ret
end
