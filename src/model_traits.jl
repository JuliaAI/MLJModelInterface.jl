## OVERLOADING TRAIT DEFAULTS RELEVANT TO MODELS

const StatTraits = StatisticalTraits

StatTraits.docstring(M::Type{<:MLJType}) = name(M)

StatTraits.prediction_type(::Type{<:Deterministic}) = :deterministic
StatTraits.prediction_type(::Type{<:Probabilistic}) = :probabilistic
StatTraits.prediction_type(::Type{<:Interval}) = :interval

# implementation is deferred as it requires methodswith which depends upon
# InteractiveUtils which we don't want to bring here as a dependency
# (even if it's stdlib).
implemented_methods(M::Type) = implemented_methods(get_interface_mode(), M)
implemented_methods(model::Model) = implemented_methods(typeof(model))
implemented_methods(::LightInterface, M) = errlight("implemented_methods")

# By default, a model is considered supervised
# TODO: add nice error message when missing y
StatTraits.abstract_type(::Type{<:Model}) = Model

for M in ABSTRACT_MODEL_SUBTYPES
    @eval(StatTraits.abstract_type(::Type{<:$M}) = $M)
end

function StatTraits.fit_data_scitype(M::Model)
    I = input_scitype(M)
    T = target_scitype(M)
    if supports_weights(M)
        W = AbstractVector{Union{Continuous,Count}} # weight scitype
        return Union{ret,Tuple{I,T,W}}
    elseif supports_class_weights(M)
        W = AbstractDict{Finite,Union{Continuous,Count}}
        return Union{ret,Tuple{I,T,W}}
    end
    return is_supervised(M) ? Tuple{I,T} : Tuple{I}
end

input_tuple(T) = Tuple{T,Union{},Union{}}
has_unsupervised(T::Type{<:Model}) = Base.hasmethod(fit, input_tuple(T))
has_supervised(T::Type{<:Model}) = Base.hasmethod(fit, input_tuple(T), (:y,))
has_weights(T::Type{<:Model}) = Base.hasmethod(fit, input_tuple(T), (:w,))

function StatTraits.is_supervised(M::Model)
    T = typeof(M)
    has_supervised(T)
end

function StatTraits.supports_weights(M::Model)
    T = typeof(M)
    has_weights(T)
end

# is_static(M::Model) = !has_unsupervised(typeof(M))

StatTraits.transform_scitype(M::Model) = output_scitype(M)
StatTraits.predict_scitype(M::Model) = target_scitype(M)
StatTraits.predict_scitype(M::Probabilistic) = _density(target_scitype(M))
StatTraits.inverse_transform_scitype(M::Type{<:Model}) = input_scitype(M)

##  `predict_scitype` FALLBACKS FOR probabilistic predictions
# This seems less than ideal but should reduce the number of `Unknown`
# in `prediction_type` for models which, historically, have not
# implemented the trait.

_density(::Any) = Unknown

for T in [:Continuous, :Count, :Textual]
    eval(
        quote
            function _density(::Type{AbstractArray{$T,D}}) where {D}
                return AbstractArray{Density{$T},D}
            end
        end
    )
end

for T in [:Finite, :Multiclass, :OrderedFactor, :Infinite, :Continuous, :Count, :Textual]
    eval(
        quote
            function _density(::Type{AbstractArray{<:$T,D}}) where {D}
                return AbstractArray{Density{<:$T},D}
            end

            _density(::Type{Table($T)}) = Table(Density{$T})
        end
    )
end

for T in [:Finite, :Multiclass, :OrderedFactor]
    eval(
        quote
            function _density(::Type{AbstractArray{<:$T{N},D}}) where {N,D}
                return AbstractArray{Density{<:$T{N}},D}
            end

            function _density(::Type{AbstractArray{$T{N},D}}) where {N,D}
                return AbstractArray{Density{$T{N}},D}
            end

            _density(::Type{Table($T{N})}) where {N} = Table(Density{$T{N}})
        end
    )
end

