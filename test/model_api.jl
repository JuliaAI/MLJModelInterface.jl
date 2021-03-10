@mlj_model mutable struct APIx0 <: Supervised
    f0::Int
end
@mlj_model mutable struct APIx0b <: Supervised
    f0::Int
end

mutable struct APIx1 <: Static end

@testset "selectrows(model, data...)" begin
    X = (x1 = [2, 4, 6],)
    y = [10.0, 20.0, 30.0]
    @test selectrows(APIx0(), 2:3, X, y) == ((x1 = [4, 6],), [20.0, 30.0])
end

@testset "fit-x" begin
    m0 = APIx0(f0=1)
    m1 = APIx0b(f0=3)
    # no weight support: fallback
    M.fit(m::APIx0, v::Int, X, y) = (5, nothing, nothing)
    @test fit(m0, 1, randn(2), randn(2), 5) == (5, nothing, nothing)
    # with weight support: use
    M.fit(m::APIx0b, v::Int, X, y, w) = (7, nothing, nothing)
    @test fit(m1, 1, randn(2), randn(2), 5) == (7, nothing, nothing)
    # default fitted params
    @test M.fitted_params(m1, 7) == (fitresult=7,)
    # static
    s1 = APIx1()
    @test fit(s1, 1, 0) == (nothing, nothing, nothing)

    #update fallback = fit
    @test update(m0, 1, 5, nothing, randn(2), 5) == (5, nothing, nothing)
end

struct DummyUnivariateFinite end

mutable struct UnivariateFiniteFitter <: Probabilistic end

@testset "models fitting a distribution to data" begin

    function MLJModelInterface.fit(model::UnivariateFiniteFitter,
                                   verbosity::Int, X, y)

        fitresult = DummyUnivariateFinite()
        report = nothing
        cache = nothing

        verbosity > 0 && @info "Fitted a $fitresult"

        return fitresult, cache, report
    end

    MLJModelInterface.predict(model::UnivariateFiniteFitter,
                          fitresult,
                          X) = fill(fitresult, length(X))

    MLJModelInterface.input_scitype(::Type{<:UnivariateFiniteFitter}) =
        Nothing
    MLJModelInterface.target_scitype(::Type{<:UnivariateFiniteFitter}) =
        AbstractVector{<:Finite}

    y =categorical(collect("aabbccaa"))
    X = nothing
    model = UnivariateFiniteFitter()
    fitresult, cache, report = MLJModelInterface.fit(model, 1, X, y)

    @test cache == nothing
    @test report == nothing

    ytest = y[1:3]
    yhat = predict(model, fitresult, fill(nothing, 3))
    @test yhat == fill(DummyUnivariateFinite(), 3)

end
