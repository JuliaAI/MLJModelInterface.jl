using Test
using MLJModelInterface

struct Opaque <: Model
    a::Int
end

struct Transparent <: Model
    A::Int
    B::Opaque
end

MLJModelInterface.istransparent(::Transparent) = true

struct Dummy <: Model
    t::Transparent
    o::Opaque
    n::Integer
end


@testset "params method" begin

    t= Transparent(6, Opaque(5))
    m = Dummy(t, Opaque(7), 42)

    @test params(m) == (
        t = (
            A = 6,
            B = (
                a = 5,
            )
        ),
        o = (
            a = 7,
        ),
        n = 42
    )
    @test flat_params(m) == Dict(
        "o__a" => 7,
        "t__A" => 6,
        "t__B__a" => 5,
        "n" => 42
    )
end
true
