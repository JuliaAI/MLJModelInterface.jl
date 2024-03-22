## OVERLOADING TRAIT DEFAULTS RELEVANT TO MODELS

# unexported aliases:
const Detector = Union{SupervisedDetector, UnsupervisedDetector}
const ProbabilisticDetector = Union{
    ProbabilisticSupervisedDetector,
    ProbabilisticUnsupervisedDetector
}
const DeterministicDetector = Union{
    DeterministicSupervisedDetector,
    DeterministicUnsupervisedDetector
}

const StatTraits = StatisticalTraits

function StatTraits.docstring(M::Type{<:Model})
    docstring = Base.Docs.doc(M) |> string
    if occursin("No documentation found", docstring)
        docstring = synthesize_docstring(M)
    end
    return docstring
end

StatTraits.is_supervised(::Type{<:Supervised}) = true
StatTraits.is_supervised(::Type{<:SupervisedAnnotator}) = true

StatTraits.prediction_type(::Type{<:Deterministic}) = :deterministic
StatTraits.prediction_type(::Type{<:Probabilistic}) = :probabilistic
StatTraits.prediction_type(::Type{<:Interval}) = :interval
StatTraits.prediction_type(::Type{<:ProbabilisticSet}) = :probabilistic_set
StatTraits.prediction_type(::Type{<:ProbabilisticDetector}) = :probabilistic
StatTraits.prediction_type(::Type{<:DeterministicDetector}) = :deterministic

function StatTraits.target_scitype(::Type{<:ProbabilisticDetector})
    return AbstractVector{<:Union{Missing,OrderedFactor{2}}}
end

function StatTraits.target_scitype(::Type{<:DeterministicDetector})
    return AbstractVector{<:Union{Missing, OrderedFactor{2}}}
end

# implementation is deferred as it requires methodswith which depends upon
# InteractiveUtils which we don't want to bring here as a dependency
# (even if it's stdlib).
implemented_methods(M::Type) = implemented_methods(get_interface_mode(), M)
implemented_methods(model) = implemented_methods(typeof(model))
implemented_methods(::LightInterface, M) = errlight("implemented_methods")

for M in ABSTRACT_MODEL_SUBTYPES
    @eval(StatTraits.abstract_type(::Type{<:$M}) = $M)
end

# helper to determine the scitype of supervised models
function supervised_fit_data_scitype(M)
    I = input_scitype(M)
    T = target_scitype(M)
    ret = Tuple{I, T}
    if supports_weights(M)
        W = AbstractVector{<:Union{Continuous, Count}} # weight scitype
        return Union{ret, Tuple{I, T, W}}
    elseif supports_class_weights(M)
        W = Any
        return Union{ret, Tuple{I, T, W}}
    end
    return ret
end

StatTraits.fit_data_scitype(M::Type{<:Unsupervised}) = Tuple{input_scitype(M)}
StatTraits.fit_data_scitype(::Type{<:Static}) = Tuple{}
StatTraits.fit_data_scitype(M::Type{<:Supervised}) = supervised_fit_data_scitype(M)

# In special case of `UnsupervisedAnnotator`, we allow the target
# as an optional argument to `fit` (that is ignored) so that the
# `machine` constructor will accept it as a valid argument, which
# then enables *evaluation* of the detector with labeled data:
function StatTraits.fit_data_scitype(M::Type{<:UnsupervisedAnnotator})
    return Union{Tuple{input_scitype(M)}, supervised_fit_data_scitype(M)}
end

function StatTraits.fit_data_scitype(M::Type{<:SupervisedAnnotator})
    return supervised_fit_data_scitype(M)
end

StatTraits.transform_scitype(M::Type{<:Unsupervised}) = output_scitype(M)
StatTraits.inverse_transform_scitype(M::Type{<:Unsupervised}) = input_scitype(M)

function StatTraits.predict_scitype(
    M::Type{<:Union{Deterministic, DeterministicDetector}}
)
    return target_scitype(M)
end

## FALLBACKS FOR `predict_scitype` FOR `Probabilistic` and
## `ProbabilisticDetector` MODELS

# This seems less than ideal but should reduce the number of `Unknown`
# in `prediction_type` for models which, historically, have not
# implemented the trait.

# Actually, this has proven more trouble than it's worth and should be removed in 2.0.
# See https://discourse.julialang.org/t/deconstructing-unionall-types/108328 to appreciate
# some of the complications.

function StatTraits.predict_scitype(
    M::Type{<:Union{Probabilistic, ProbabilisticDetector}}
)
    return _density(target_scitype(M))
end

const SCALAR_SCITYPES_EXS =
    [:Finite, :Multiclass, :OrderedFactor, :Infinite, :Continuous, :Count, :Textual]

const SCALAR_SCITYPES =
    eval.([:Finite, :Multiclass, :OrderedFactor, :Infinite, :Continuous, :Count, :Textual])

function _density(t)
    for T in SCALAR_SCITYPES
        t == AbstractVector{<:T} && return AbstractVector{Density{<:T}}
        t == AbstractMatrix{<:T} && return AbstractMatrix{Density{<:T}}
        t == Table(T) && return Table{<:AbstractVector{<:Density{<:T}}}
    end
    for T in  [Finite, Multiclass, OrderedFactor]
        t == Table(T{2}) && return Table(Density{<:T{2}})
    end
    return Unknown
end

for T in SCALAR_SCITYPES_EXS
        quote
            _density(::Type{<:AbstractArray{W, D}}) where {W<:$T, D} =
                AbstractArray{Density{W}, D}
            _density(::Type{<:Table{<:AbstractVector{W}}}) where W<:$T =
                Table(Density{W})
        end |> eval
end
