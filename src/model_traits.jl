## MODEL TRAITS

# model trait names:
const MODEL_TRAITS = [
    :input_scitype, :output_scitype, :target_scitype,
    :is_pure_julia, :package_name, :package_license,
    :load_path, :package_uuid, :package_url,
    :is_wrapper, :supports_weights, :supports_online,
    :docstring, :name, :is_supervised,
    :prediction_type, :implemented_methods, :hyperparameters,
    :hyperparameter_types, :hyperparameter_ranges]

const SUPERVISED_TRAITS = setdiff(MODEL_TRAITS, [:output_scitype])

const UNSUPERVISED_TRAITS = setdiff(MODEL_TRAITS,
    [:target_scitype, :prediction_type, :supports_weights])

for trait in MODEL_TRAITS
    ex = quote
        $trait(x) = $trait(typeof(x))
    end
    MLJModelInterface.eval(ex)
end

# fallback trait declarations:
input_scitype(::Type)          = Unknown
output_scitype(::Type)         = Unknown
target_scitype(::Type)         = Unknown  # used for measures too
is_pure_julia(::Type)          = false
package_name(::Type)           = "unknown"
package_license(::Type)        = "unknown"
load_path(::Type)              = "unknown"
package_uuid(::Type)           = "unknown"
package_url(::Type)            = "unknown"
is_wrapper(::Type)             = false
supports_online(::Type)        = false
supports_weights(::Type)       = false  # used for measures too
hyperparameter_ranges(T::Type) = Tuple(fill(nothing, length(fieldnames(T))))
docstring(M::Type)             = string(M)
docstring(M::Type{<:MLJType})  = name(M)
docstring(M::Type{<:Model})    = "$(name(M)) from $(package_name(M)).jl.\n" *
                                 "[Documentation]($(package_url(M)))."
# "derived" traits:
name(M::Type)            = string(M)
name(M::Type{<:MLJType}) = split(string(coretype(M)), '.')[end] |> String

is_supervised(::Type)                    = false
is_supervised(::Type{<:Supervised})      = true
prediction_type(::Type)                  = :unknown # used for measures too
prediction_type(::Type{<:Deterministic}) = :deterministic
prediction_type(::Type{<:Probabilistic}) = :probabilistic
prediction_type(::Type{<:Interval})      = :interval
implemented_methods(M::Type{<:MLJType})  = getfield.(methodswith(M), :name)
hyperparameters(M::Type)                 = fieldnames(M)
hyperparameter_types(M::Type)            = string.(fieldtypes(M))
