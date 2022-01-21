using StatisticalTraits

@mlj_model mutable struct S1 <: Model end

function M.fit(m::S1, X, verbosity; y) end

@mlj_model mutable struct U1 <: Model end

@mlj_model mutable struct D1 <: Deterministic
    a::Int = 1::(_ > 0)
end

@mlj_model mutable struct P1 <: Probabilistic
    a::Int = 1::(_ > 0)
end

@mlj_model mutable struct I1 <: Interval end

foo(::P1) = 0
bar(::P1) = nothing

@testset "traits" begin
    ms = S1()
    mu = U1()
    md = D1()
    mp = P1()
    mi = I1()

    @test input_scitype(ms) == Unknown
    @test output_scitype(ms) == Unknown
    @test target_scitype(ms) == Unknown
    @test is_pure_julia(ms) == false

    @test package_name(ms) == "unknown"
    @test package_license(ms) == "unknown"
    @test load_path(ms) == "unknown"
    @test package_uuid(ms) == "unknown"
    @test package_url(ms) == "unknown"

    @test is_wrapper(ms) == false
    @test supports_online(ms) == false
    @test supports_weights(ms) == false
    @test iteration_parameter(ms) === nothing

    @test hyperparameter_ranges(md) == (nothing,)

    @test docstring(ms) == "S1"
    @test name(ms) == "S1"

    @test is_supervised(ms)
    @test !is_supervised(mu)
    @test prediction_type(ms) == :unknown
    @test prediction_type(md) == :deterministic
    @test prediction_type(mp) == :probabilistic
    @test prediction_type(mi) == :interval

    @test hyperparameters(md) == (:a,)
    @test hyperparameter_types(md) == ("Int64",)

    # implemented methods is deferred
    setlight()
    @test_throws M.InterfaceError implemented_methods(mp)

    setfull()

    function M.implemented_methods(::FI, M::Type{<:MLJType})
        return getfield.(methodswith(M), :name)
    end

    @test Set(implemented_methods(mp)) == Set([:clean!, :bar, :foo])
end

module Fruit

import MLJModelInterface.MLJType

struct Banana <: MLJType end

end

import .Fruit

@testset "extras" begin
    @test docstring(Float64) == "Float64"
    @test docstring(Fruit.Banana) == "Banana"
end

@testset "`_density` - helper for predict_scitype fallback" begin
    for T in [Continuous, Count, Textual]
        @test ==(
            M._density(AbstractArray{T,3}),
            AbstractArray{Density{T},3}
        )
    end

    for T in [Finite,
        Multiclass,
        OrderedFactor,
        Infinite,
        Continuous,
        Count,
        Textual]
        @test ==(
            M._density(AbstractVector{<:T}),
            AbstractVector{Density{<:T}}
        )
        @test M._density(Table(T)) == Table(Density{T})
    end

    for T in [Finite, Multiclass, OrderedFactor]
        @test ==(
            M._density(AbstractArray{<:T{2},3}),
            AbstractArray{Density{<:T{2}},3}
        )
        @test ==(
            M._density(AbstractArray{T{2},3}),
            AbstractArray{Density{T{2}},3}
        )
        @test M._density(Table(T{2})) == Table(Density{T{2}})
    end
end

@mlj_model mutable struct P2 <: Probabilistic end
function MLJModelInterface.fit(::P2, X, verbosity; y) end
M.target_scitype(::Type{<:P2}) = AbstractVector{<:Multiclass}
M.input_scitype(::Type{<:P2}) = Table(Continuous)

@mlj_model mutable struct U2 <: Model end
M.output_scitype(::Type{<:U2}) = AbstractVector{<:Multiclass}
M.input_scitype(::Type{<:U2}) = Table(Continuous)

# TODO: STATIC
# @mlj_model mutable struct S2 <: Model end
# M.output_scitype(::Type{<:S2}) = AbstractVector{<:Multiclass}
# M.input_scitype(::Type{<:S2}) = Table(Continuous)

@testset "operation scitypes" begin
    @test predict_scitype(P2()) == AbstractVector{Density{<:Multiclass}}
    @test transform_scitype(P2()) == Unknown
    @test transform_scitype(U2()) == AbstractVector{<:Multiclass}
    @test inverse_transform_scitype(U2()) == Table(Continuous)
    @test predict_scitype(U2()) == Unknown
    # TODO: STATIC
    # @test transform_scitype(S2()) == AbstractVector{<:Multiclass}
    # @test inverse_transform_scitype(S2()) == Table(Continuous)
end

@testset "abstract_type, fit_data_scitype" begin
    @test abstract_type(P2()) == Probabilistic
    @test abstract_type(S1()) == Model
    @test abstract_type(U1()) == Model
    @test abstract_type(D1()) == Deterministic
    @test abstract_type(P1()) == Probabilistic

    @test ==(
        fit_data_scitype(P2()),
        Tuple{Table(Continuous),AbstractVector{<:Multiclass}}
    )
    @test fit_data_scitype(U2()) == Tuple{Table(Continuous)}
    # TODO: STATIC
    # @test fit_data_scitype(S2()) == Tuple{}
end

true
