using Test
using MLJModelInterface

struct Opaque
    a::Int
end

struct Transparent
    A::Int
    B::Opaque
end

MLJModelInterface.istransparent(::Transparent) = true

struct Dummy <:MLJType
    t::Transparent
    o::Opaque
    n::Integer
end

@testset "params method" begin

    t= Transparent(6, Opaque(5))
    m = Dummy(t, Opaque(7), 42)

    @test params(m) == (t = (A = 6,
                             B = Opaque(5)),
                        o = Opaque(7),
                        n = 42)
end
true
