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

M.metadata_model(
    APIx0,
    supports_training_losses = true,
    reports_feature_importances = true,
)

dummy_losses = [1.0, 2.0, 3.0]
M.training_losses(::APIx0, report) = report
M.feature_importances(::APIx0, fitresult, report) = [:a=>0, :b=>0]

@testset "fit-x" begin
    m0 = APIx0(f0=1)
    m1 = APIx0b(f0=3)
    # no weight support: fallback
    M.fit(m::APIx0, v::Int, X, y) = (5, nothing, dummy_losses)
    @test fit(m0, 1, randn(2), randn(2), 5) == (5, nothing, dummy_losses)
    # with weight support: use
    M.fit(m::APIx0b, v::Int, X, y, w) = (7, nothing, nothing)
    @test fit(m1, 1, randn(2), randn(2), 5) == (7, nothing, nothing)
    # default fitted params
    @test M.fitted_params(m1, 7) == (fitresult=7,)
    # default iteration_parameter
    @test M.training_losses(m0, nothing) === nothing
    # static
    s1 = APIx1()
    @test fit(s1, 1, 0) == (nothing, nothing, nothing)

    # update fallback = fit
    @test update(m0, 1, 5, nothing, randn(2), 5) == (5, nothing, dummy_losses)

    # training losses:
    f, c, r = MLJModelInterface.fit(m0, 1, rand(2), rand(2))
    @test M.training_losses(m0, r) == dummy_losses

    # training losses:
    f, c, r = MLJModelInterface.fit(m0, 1, rand(2), rand(2))
    @test M.training_losses(m0, r) == dummy_losses

    # feature_importances
    f, c, r = MLJModelInterface.fit(m0, 1, rand(2), rand(2))
    @test MLJModelInterface.feature_importances(m0, f, r) == [:a=>0, :b=>0]
end

struct DummyUnivariateFinite end

mutable struct UnivariateFiniteFitter <: Probabilistic end

@testset "models fitting a distribution to data" begin
    MMI = MLJModelInterface

    function MMI.fit(model::UnivariateFiniteFitter, verbosity::Int, X, y)
        fitresult = DummyUnivariateFinite()
        report = nothing
        cache = nothing

        verbosity > 0 && @info "Fitted a $fitresult"

        return fitresult, cache, report
    end

    function MMI.predict(model::UnivariateFiniteFitter, fitresult, X)
        return fill(fitresult, length(X))
    end

    MMI.input_scitype(::Type{<:UnivariateFiniteFitter}) = Nothing

    MMI.target_scitype(::Type{<:UnivariateFiniteFitter}) = AbstractVector{<:Finite}

    y = categorical(collect("aabbccaa"))
    X = nothing
    model = UnivariateFiniteFitter()
    fitresult, cache, report = MMI.fit(model, 1, X, y)

    @test cache === nothing
    @test report === nothing

    ytest = y[1:3]
    yhat = predict(model, fitresult, fill(nothing, 3))
    @test yhat == fill(DummyUnivariateFinite(), 3)

end

@testset "fallback for `report()` method" begin
    report_given_method =
        OrderedCollections.OrderedDict(
            :predict=>(y=7,),
            :fit=>(x=1, z=3),
            :transform=>nothing,
        )
    @test MLJModelInterface.report(APIx0(f0=1), report_given_method) ==
        (x=1, z=3, y=7)

    report_given_method =
        OrderedCollections.OrderedDict(
            :predict=>(y=7,),
            :fit=>(y=1, z=3),
            :transform=>nothing,
        )
    @test MLJModelInterface.report(APIx0(f0=1), report_given_method) ==
        (y=1, z=3, predict=(y=7,))

    @test MLJModelInterface.report(
        APIx0(f0=1),
        OrderedCollections.OrderedDict(:fit => nothing, :transform => NamedTuple()),
    ) == nothing

    @test MLJModelInterface.report(
        APIx0(f0=1),
        OrderedCollections.OrderedDict(:fit => 42),
    ) == 42


end
