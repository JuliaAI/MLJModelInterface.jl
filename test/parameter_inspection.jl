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

struct Dummy <: MLJType
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
            B = Opaque(5)
        ),
        o = Opaque(7),
        n = 42
    )
end

struct ChildModel <: Model
    r::Int
    s
end

struct ParentModel <: Model
    x::Int
    y::String
    first_child::ChildModel
    second_child::ChildModel
end

struct Missy <: Model end

@testset "flat_params method" begin

    m = ParentModel(1, "parent", ChildModel(2, "child1"),
        ChildModel(3, Missy()))

    @test MLJModelInterface.flat_params(m) == (
        x = 1,
        y = "parent",
        first_child__r = 2,
        first_child__s = "child1",
        second_child__r = 3,
        second_child__s = Missy()
    )
end
true
