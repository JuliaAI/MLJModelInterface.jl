## MODEL TRAITS

# model trait names:

StatisticalTraits.docstring(M::Type{<:MLJType}) = name(M)
StatisticalTraits.docstring(M::Type{<:Model}) =
    "$(name(M)) from $(package_name(M)).jl.\n" *
    "[Documentation]($(package_url(M)))."

StatisticalTraits.is_supervised(::Type{<:Supervised})      = true
StatisticalTraits.prediction_type(::Type{<:Deterministic}) = :deterministic
StatisticalTraits.prediction_type(::Type{<:Probabilistic}) = :probabilistic
StatisticalTraits.prediction_type(::Type{<:Interval})      = :interval

# implementation is deferred as it requires methodswith which depends upon
# InteractiveUtils which we don't want to bring here as a dependency
# (even if it's stdlib).
implemented_methods(M::Type) = implemented_methods(get_interface_mode(), M)
implemented_methods(model) = implemented_methods(typeof(model))
implemented_methods(::LightInterface, M) = errlight("implemented_methods")
