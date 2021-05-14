using Random
using MLJModelInterface
using Test

mutable struct Foo <: MLJType
    rng::AbstractRNG
    x::Int
    y::Int
end

mutable struct Bar{names} <: MLJType
    rng::AbstractRNG
    v::Tuple{Int,Int}
    Bar{names}(rng, x, y) where names =
        new{names}(rng, (x, y))
end

Bar(rng, x, y) = Bar{(:x, :y)}(rng, x, y)

# overload `getproperty` so that components of `v` are accessed with
# the names given in `names` (which will be (:x, :y) when using
# the above constructor):
Base.propertynames(::Bar{names}) where names = (:rng, names...)
function Base.getproperty(b::Bar{names}, name::Symbol) where names
    name === :rng && return getfield(b, :rng)
    v = getfield(b, :v)
    name === names[1] && return v[1]
    return v[2]
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

mutable struct Sub
    x::Int
end

mutable struct Deep
    x::Int
    s::Union{Sub,Int}
end

mutable struct Super2 <: MLJType
    sub::Sub
    z::Int
end

MLJModelInterface.deep_properties(::Type{<:Super2}) = (:sub,)

@testset "_equal_to_depth_one" begin
    d1 = Deep(1, 2)
    d2 = Deep(1, 2)
    @test MLJModelInterface._equal_to_depth_one(d1, d2)
    d2.x = 3
    @test !MLJModelInterface._equal_to_depth_one(d1, d2)

    d1 = Deep(1, Sub(2))
    d2 = Deep(1, Sub(2))
    @test !MLJModelInterface._equal_to_depth_one(d1, d2)
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
    @test p1 == p2
    p2.y = [1,2]
    @test !(p1 == p2)

    # test of "deep" properties
    s1 = Super2(Sub(1), 2)
    s2 = Super2(Sub(1), 2)
    @test s1 == s2

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

@testset "isrepresented" begin
    m = Foo(MersenneTwister(7), 1, 2)
    m2 = Foo(MersenneTwister(8), 1, 2)
    n = Foo(MersenneTwister(7), 3, 2)
    p = Foo(MersenneTwister(9), 4, 5)
    models = [m, n]

    @test isrepresented(m, nothing) == false
    @test isrepresented(m, models)
    @test isrepresented(m2, models)
    @test !isrepresented(p, models)
end

true
