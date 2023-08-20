mutable struct A0 <: Model
    f0::Int
end

@testset "clean!" begin
    a = A0(0)
    @test clean!(a) == ""
    function M.clean!(a::A0)
        warn = ""
        if a.f0 < 0
            warn *= "Field a is negative, resetting to 5."
            a.f0 = 5
        end
        return warn
    end
    a = A0(-2)
    @test clean!(a) == "Field a is negative, resetting to 5."
    @test a.f0 == 5
end

# No type, no default
@mlj_model mutable struct A1
    a
end

@testset "@mlj-1" begin
    a = A1()
    @test ismissing(a.a)
    a.a = 5
    @test a.a == 5
end

# No type, with default
@mlj_model mutable struct A1b
    a = 5
end

@testset "@mlj-2" begin
    a = A1b()
    @test a.a == 5
    a.a = "hello"
    @test a.a == "hello"
end

# If a type is given but no default value is given, then the macro tries to fill
# a default value; either 0 if it's a Number type, or an empty string and otherwise fails.
@mlj_model mutable struct A1c
    a::Int
end

@testset "@mlj-3" begin
    a = A1c()
    @test a.a == 0
    a = A1c(a=7)
    @test a.a == 7
    @test_throws InexactError A1c(a=5.3)
    @test_throws MethodError A1c(a="hello")
end

# Type is given and default is given
@mlj_model mutable struct A1d
    a::Int = 5
end

@testset "@mlj-4" begin
    a = A1d()
    @test a.a == 5
    a = A1d(a=7)
    @test a.a == 7
end

# No type is given but a default and constraint
@mlj_model mutable struct A1e
    a = 5::(_ > 0)
end

@testset "@mlj-5" begin
    a = A1e()
    @test a.a == 5
    a = A1e(a=7)
    @test a.a == 7
    @test @test_logs (:warn, "Constraint `model.a > 0` failed; using default: a=5.") A1e(a=-1).a==5
    a = A1e(a=7.5)
    @test a.a == 7.5
end

# Type is given with default and constraint
@mlj_model mutable struct A1f
    a::Int = 5::(_ > 0)
end

@testset "@mlj-6" begin
    a = A1f()
    @test a.a == 5
    a = A1f(a=7)
    @test a.a == 7
    @test_throws InexactError A1f(a=7.5)
    @test @test_logs (:warn, "Constraint `model.a > 0` failed; using default: a=5.") A1f(a=-1).a==5
end

abstract type FooBar end
@mlj_model mutable struct B1a <: FooBar
    a::Symbol = :auto::(_ in (:auto, :semi))
end

@testset "@mlj-7" begin
    b = B1a()
    @test b.a == :auto
    b = B1a(a=:semi)
    @test b.a == :semi
    @test @test_logs (:warn, "Constraint `model.a in (:auto, :semi)` failed; using default: a=:auto.") B1a(a=:autos).a == :auto
    @test_throws MethodError B1a(b="blah")
end

# == dependence on other types

@mlj_model mutable struct B1b
    a::SemiMetric = Euclidean()::(_ isa Metric)
end

@mlj_model mutable struct B1c
    a::SemiMetric = Euclidean()
end

@testset "@mlj-dist" begin
    @test B1b().a isa Euclidean
    @test @test_logs (:warn, "Constraint `model.a isa Metric` failed; using default: a=Euclidean().") B1b(a=BhattacharyyaDist()).a isa Euclidean
    @test B1c().a isa Euclidean
end

# Implicit defaults
@mlj_model mutable struct Ca
    a::String
end

@mlj_model mutable struct Cb
    a::Any
end

@mlj_model mutable struct Cc
    a::Union{Nothing,Int}
end

@mlj_model mutable struct Cd
    a::Union{Missing,Int}
end

@testset "@mlj-10" begin
    @test Ca().a == ""
    @test Cb().a === missing
    @test Cc().a === nothing
    @test Cd().a === missing
end

@testset "Expression defaults" begin
    # Should work with and without constraint:
    @mlj_model mutable struct Foo1
        a::Vector{Int} = [1, 2, 3]
    end
    @test Foo1().a == [1, 2, 3]
    @mlj_model mutable struct Foo2
        a::Vector{Int} = [1, 2, 3]::(true)
    end
    @test Foo2().a == [1, 2, 3]

    # Constraints applied
    @mlj_model mutable struct Foo3
        a::Vector{Int} = [1, 2, 3]::(all(>(0), _))
    end
    @test redirect_stderr(devnull) do
        Foo3(; a = [-1]).a == [1, 2, 3]
    end

    # Negative number:
    @mlj_model mutable struct Foo4
        a::Float64 = -1.0
    end
    @test Foo4().a === -1.0
    @mlj_model mutable struct Foo5
        a::Float64 = (-1.0)::(true)
    end
    @test Foo5().a == -1.0

end
