using Random
using MLJModelInterface
using Test

mutable struct Foo <: MLJType
    rng::AbstractRNG
    x::Int
    y::Int
end

mutable struct Bar <: MLJType
    rng::AbstractRNG
    x::Int
    y::Int
end

mutable struct Super <: MLJType
    sub::Foo
    z::Int
end

mutable struct Partial <: MLJType
    x::Int
    y::Vector{Int}
    Partial(x) = new(x)
end

@testset "equality for MLJType" begin
    f1 = Foo(MersenneTwister(7), 1, 2)
    f2 = Foo(MersenneTwister(8), 1, 2)

    @test f1.rng != f2.rng
    @test f1 == f2
    f1.x = 10
    @test f1 != f2
    b = Bar(MersenneTwister(7), 1, 2)
    @test f2 != b

    @test is_same_except(f1, f2, :x)
    f1.y = 20
    @test f1 != f2
    @test is_same_except(f1, f2, :x, :y)

    f1 = Foo(MersenneTwister(7), 1, 2)
    f2 = Foo(MersenneTwister(8), 1, 2)
    s1 = Super(f1, 20)
    s2 = Super(f2, 20)
    @test s1 == s2
    s2.sub.x = 10
    @test f1 != f2

    @test !(f1 == Super(f1, 4))

    @test !(isequal(Foo(MersenneTwister(1), 1, 2),
                    Foo(MersenneTwister(1), 1, 2)))

    p1 = Partial(1)
    p2 = Partial(1)
    p2.y = [1,2]
    @test !(p1 == p2)

end

@testset "in(x, collection) for MLJType" begin
    f1 = Foo(MersenneTwister(7), 1, 2)
    f2 = Foo(MersenneTwister(7), 1, 2)
    f3 = Super(f1, 20)

    tv = (f1, f3)
    tk = (f2, f3)
    tw = (f3, f3)
    v = [tv...]
    k = [tk...]
    w = [tw...]
    @test f1 in tv
    @test !(f1 in tk)
    @test !(f1 in tw)
    @test f1 in v
    @test !(f1 in k)
    @test !(f1 in w)
    @test f1 in Set(v)
    @test !(f1 in Set(k))
    @test !(f1 in Set(w))

end

true
