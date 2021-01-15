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

StatTraits.docstring(M::Type{<:MLJType}) = name(M)
StatTraits.docstring(M::Type{<:Model}) =
    "$(name(M)) from $(package_name(M)).jl.\n" *
    "[Documentation]($(package_url(M)))."

StatTraits.is_supervised(::Type{<:Supervised})      = true
StatTraits.prediction_type(::Type{<:Deterministic}) = :deterministic
StatTraits.prediction_type(::Type{<:Probabilistic}) = :probabilistic
StatTraits.prediction_type(::Type{<:Interval})      = :interval

# implementation is deferred as it requires methodswith which depends upon
# InteractiveUtils which we don't want to bring here as a dependency
# (even if it's stdlib).
implemented_methods(M::Type) = implemented_methods(get_interface_mode(), M)
implemented_methods(model) = implemented_methods(typeof(model))
implemented_methods(::LightInterface, M) = errlight("implemented_methods")
