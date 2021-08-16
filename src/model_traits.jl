## OVERLOADING TRAIT DEFAULTS RELEVANT TO MODELS

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

for M in ABSTRACT_MODEL_SUBTYPES
    @eval(StatisticalTraits.abstract_type(::Type{<:$M}) = $M)
end

StatisticalTraits.training_scitype(M::Type{<:Model}) = input_scitype(M)
StatisticalTraits.training_scitype(::Type{<:Static}) = Tuple{}
function StatisticalTraits.training_scitype(M::Type{<:Supervised})
    I = input_scitype(M)
    T = target_scitype(M)
    ret = Tuple{I,T}
    if supports_weights(M)
        W = AbstractVector{Union{Continuous,Count}} # weight scitype
        return Union{ret,Tuple{I,T,W}}
    elseif supports_class_weights(M)
        W = AbstractDict{Finite,Union{Continuous,Count}}
        return Union{ret,Tuple{I,T,W}}
    end
    return ret
end

StatisticalTraits.transform_scitype(M::Type{<:Unsupervised}) =
    output_scitype(M)

StatisticalTraits.inverse_transform_scitype(M::Type{<:Unsupervised}) =
    input_scitype(M)

StatisticalTraits.predict_scitype(M::Type{<:Deterministic}) = target_scitype(M)


## FALLBACKS FOR `predict_scitype` FOR `Probabilistic` MODELS

# This seems less than ideal but should reduce the number of `Unknown`
# in `prediction_type` for models which, historically, have not
# implemented the trait.

StatisticalTraits.predict_scitype(M::Type{<:Probabilistic}) =
    _density(target_scitype(M))

_density(::Any) = Unknown
for T in [:Continuous, :Count, :Textual]
    eval(quote
         _density(::Type{AbstractArray{$T,D}}) where D =
         AbstractArray{Density{$T},D}
         end)
end

for T in [:Finite,
          :Multiclass,
          :OrderedFactor,
          :Infinite,
          :Continuous,
          :Count,
          :Textual]
    eval(quote
         _density(::Type{AbstractArray{<:$T,D}}) where D =
         AbstractArray{Density{<:$T},D}
         _density(::Type{Table($T)}) = Table(Density{$T})
         end)
end

for T in [:Finite, :Multiclass, :OrderedFactor]
    eval(quote
         _density(::Type{AbstractArray{<:$T{N},D}}) where {N,D} =
         AbstractArray{Density{<:$T{N}},D}
         _density(::Type{AbstractArray{$T{N},D}}) where {N,D} =
         AbstractArray{Density{$T{N}},D}
         _density(::Type{Table($T{N})}) where N = Table(Density{$T{N}})
         end)
end
