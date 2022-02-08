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

# Extend `program` (an expression) to include trait definition for
# specified `trait` and type `T`.
function _extend!(program::Expr, trait::Symbol, value, T)
    if value !== nothing
        push!(program.args, quote
              MLJModelInterface.$trait(::Type{<:$T}) = $value
              end)
    end
end

const WARN_MISSING_LOAD_PATH = "No `load_path` defined. "


"""
    metadata_model(`T`; args...)

Helper function to write the metadata for a model `T`.

## Keywords

* `input_scitype=Unknown`: allowed scientific type of the input data
* `target_scitype=Unknown`: allowed scitype of the target (supervised)
* `output_scitype=Unkonwn`: allowed scitype of the transformed data (unsupervised)
* `supports_weights=false`: whether the model supports sample weights
* `supports_class_weights=false`: whether the model supports class weights
* `load_path="unknown"`: where the model is (usually `PackageName.ModelName`)

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
    supports_class_weights::Union{Nothing,Bool}=weights,
    docstring::Union{Nothing,String}=descr,
    load_path::Union{Nothing,String}=path,
)

    load_path === nothing && @warn WARN_MISSING_LOAD_PATH

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

    parentmodule(T).eval(program)
end
