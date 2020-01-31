@testset "matrix-light" begin
    setlight()
    X = ones(2,3)
    @test matrix(X) === X
    @test matrix(X, transpose=true) == ones(3,2)
    X = (1,2,3)
    @test_throws ArgumentError matrix(X)
    X = (a=[1,2,3],b=[1,2,3])
    @test_throws M.InterfaceError matrix(X)
end
@testset "matrix-full" begin
    setfull()
    M.matrix(::FI, ::Val{:table}, X; kw...) = Tables.matrix(X; kw...)
    X = (a=[1,2,3],b=[1,2,3])
    @test matrix(X) == hcat([1,2,3],[1,2,3])
end
# ------------------------------------------------------------------------
@testset "int-light" begin
    setlight()
    x = categorical([1,2,3])
    @test_throws M.InterfaceError int(x)
end
@testset "int-full" begin
    setfull()
    M.int(::FI, x::CategoricalElement; kw...) =
        CategoricalArrays.order(x.pool)[x.level]
    x = categorical(['a','b','a'])
    @test int(x[1]) == 0x01
    @test int(x[2]) == 0x02
end
# ------------------------------------------------------------------------
@testset "classes-light" begin
    setlight()
    x = categorical(['a','b','a'])
    @test_throws M.InterfaceError classes(x)
end
@testset "classes-full" begin
    setfull()
    M.classes(::FI, p::CategoricalPool) =
        [p[i] for i in invperm(CategoricalArrays.order(p))]
    M.classes(::FI, x::CategoricalElement) = classes(x.pool)
    x = categorical(['a','b','a'])
    @test classes(x[1]) == ['a', 'b']
end
# ------------------------------------------------------------------------
@testset "decoder-light" begin
    setlight()
    x = 5
    @test_throws M.InterfaceError decoder(x)
end
@testset "decoder-full" begin
    setfull()
    # toy test because I don't want to copy the decoder logic here
    M.decoder(::FI, x) = 0
    @test decoder(nothing) == 0
end
# ------------------------------------------------------------------------
@testset "table-light" begin
    setlight()
    X = ones(3,2)
    @test_throws M.InterfaceError table(X)
end
@testset "table-full" begin
    setfull()
    function M.table(::FI, A::AbstractMatrix; names=nothing)
        _names = [Symbol(:x, j) for j in 1:size(A, 2)]
        return Tables.table(A, header=_names)
    end
    X = ones(3,2)
    T = table(X)
    @test Tables.istable(T)
    @test Tables.matrix(T) == X
end
# ------------------------------------------------------------------------
@testset "nrows-light" begin
    setlight()
    X = ones(5)
    @test nrows(X) == 5
    X = ones(5,3)
    @test nrows(X) == 5
    X = ones(5,3,2)
    @test_throws ArgumentError nrows(X)
    X = (a=[4,2,1],b=[3,2,1])
    @test_throws M.InterfaceError nrows(X)
end
@testset "nrows-full" begin
    setfull()
    M.nrows(::FI, ::Val{:table}, X) = Tables.rowcount(X)
    X = (a=[4,2,1],b=[3,2,1])
    @test nrows(X) == 3
end
# ------------------------------------------------------------------------
@testset "select-light" begin
    setlight()
    X = nothing
    @test selectrows(X, 1) === nothing
    @test selectcols(X, 1) === nothing
    @test select(X, 1, 2) === nothing

    # vector
    X = ones(5)
    @test selectrows(X, 1)   == [1.0]
    @test selectrows(X, 1:2) == ones(2,)
    @test selectrows(X, :)  === X
    @test_throws ArgumentError selectcols(X, 5)
    @test_throws ArgumentError select(X, 2, 2)

    # matrix
    X = ones(5, 3)
    @test selectrows(X, 1)    == ones(1, 3)
    @test selectrows(X, 1:2)  == ones(2, 3)
    @test selectrows(X, :)   === X
    @test selectcols(X, 1)    == ones(5,)
    @test selectcols(X, 1:2)  == ones(5, 2)
    @test selectcols(X, :)   === X
    @test select(X, 1, 1)     == [1.0]
    @test select(X, 1:2, 1)   == ones(2,)
    @test select(X, 1:2, 1:2) == ones(2, 2)

    # table
    X = (x=[1,1,1],y=[2,2,2])
    @test_throws M.InterfaceError selectrows(X, 1)
    @test_throws M.InterfaceError selectcols(X, 1)

    # something else
    X = (1,2,3)
    @test_throws ArgumentError selectrows(X, 1)
    @test_throws ArgumentError selectcols(X, 1)
    @test_throws ArgumentError select(X, 1, 1)
end
# ------------------------------------------------------------------------
@testset "select-full" begin
    setfull()
    M.selectrows(::FI, ::Val{:table}, X, ::Colon) = X
    M.selectcols(::FI, ::Val{:table}, X, ::Colon) = X
    function M.selectrows(::FI, ::Val{:table}, X, r)
        r = r isa Integer ? (r:r) : r
        cols = Tables.columntable(X)
        new_cols = NamedTuple{keys(cols)}(tuple((c[r] for c in values(cols))...))
        return Tables.materializer(X)(new_cols)
    end
    function M.selectcols(::FI, ::Val{:table}, X, c::Union{Symbol,Integer})
        cols = Tables.columntable(X) # named tuple of vectors
        return cols[c]
    end
    function M.selectcols(::FI, ::Val{:table}, X, c::AbstractArray)
        cols = Tables.columntable(X) # named tuple of vectors
        newcols = project(cols, c)
        return Tables.materializer(X)(newcols)
    end
    # project named tuple onto a tuple with only specified `labels` or indices:
    project(t::NamedTuple, labels::AbstractArray{Symbol}) =
        NamedTuple{tuple(labels...)}(t)
    project(t::NamedTuple, label::Colon) = t
    project(t::NamedTuple, label::Symbol) = project(t, [label,])
    project(t::NamedTuple, indices::AbstractArray{<:Integer}) =
        NamedTuple{tuple(keys(t)[indices]...)}(tuple([t[i] for i in indices]...))
    project(t::NamedTuple, i::Integer) = project(t, [i,])

    X = (x=[1,2,3],y=[4,5,6], z=[0,0,0])
    @test selectrows(X, 1)   == (x=[1],y=[4],z=[0])
    @test selectrows(X, 1:2) == (x=[1,2],y=[4,5],z=[0,0])
    @test selectrows(X, :)  === X
    @test selectcols(X, 1)   == [1,2,3]
    @test selectcols(X, 1:2) == (x = [1, 2, 3], y = [4, 5, 6])
    @test selectcols(X, :)  === X
    @test select(X, 1, 1)    == [1]
    @test select(X, 1:2, 1)  == [1,2]
    @test select(X, :, 1)    == [1,2,3]
    @test selectcols(X, :x)  == [1,2,3]
    @test select(X, 1:2, :z) == [0,0]
end

@testset "univ-finite" begin
    setlight()
    @test_throws M.InterfaceError UnivariateFinite(Dict(2=>3,3=>4))
    @test_throws M.InterfaceError UnivariateFinite(randn(2), randn(2))
end
