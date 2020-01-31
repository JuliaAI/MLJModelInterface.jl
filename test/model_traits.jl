@mlj_model mutable struct S1 <: Supervised
end

@mlj_model mutable struct U1 <: Unsupervised
end

@mlj_model mutable struct D1 <: Deterministic
    a::Int = 1::(_ > 0)
end

@mlj_model mutable struct P1 <: Probabilistic
    a::Int = 1::(_ > 0)
end

@mlj_model mutable struct I1 <: Interval
end

foo(::P1) = 0
bar(::P1) = nothing

@testset "traits" begin
    ms = S1()
    mu = U1()
    md = D1()
    mp = P1()
    mi = I1()

    @test input_scitype(ms)  == Unknown
    @test output_scitype(ms) == Unknown
    @test target_scitype(ms) == Unknown
    @test is_pure_julia(ms)  == false

    @test package_name(ms)    == "unknown"
    @test package_license(ms) == "unknown"
    @test load_path(ms)       == "unknown"
    @test package_uuid(ms)    == "unknown"
    @test package_url(ms)     == "unknown"

    @test is_wrapper(ms)       == false
    @test supports_online(ms)  == false
    @test supports_weights(ms) == false

    @test hyperparameter_ranges(md) == (nothing,)

    @test docstring(ms) == "S1 from unknown.jl.\n[Documentation](unknown)."
    @test name(ms)      == "S1"

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
    M.implemented_methods(::FI, M::Type{<:MLJType}) =
        getfield.(methodswith(M), :name)
    @test implemented_methods(mp) == [:clean!,:bar,:foo]
end
