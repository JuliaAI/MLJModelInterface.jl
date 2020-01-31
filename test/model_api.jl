@mlj_model mutable struct APIx0 <: Supervised
    f0::Int
end
@mlj_model mutable struct APIx0b <: Supervised
    f0::Int
end

mutable struct APIx1 <: Static end

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
