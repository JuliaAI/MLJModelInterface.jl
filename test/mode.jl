@testset "light" begin
    M.set_interface_mode(M.LightInterface())
    @test M.get_interface_mode() isa M.LightInterface

    # matrix object (:other)
    X   = zeros(3, 4)
    mX  = matrix(X)
    mtX = matrix(X; transpose=true)

    @test mX === X
    @test mtX == permutedims(X)

    # :other but not matrix
    X = (1, 2, 3, 4)
    @test_throws ArgumentError matrix(X)

    # :table
    X = (x=[1,2,3], y=[1,2,3])
    @test M.vtrait(X) isa Val{:table}
    @test_throws M.InterfaceError matrix(X)
end

@testset "full" begin
    M.set_interface_mode(M.FullInterface())
    @test M.get_interface_mode() isa M.FullInterface

    M.matrix(::M.FullInterface, ::Val{:table}, X; kw...) =
        Tables.matrix(X; kw...)

    X = (x=[1,2,3], y=[1,2,3])
    mX = matrix(X)
    @test mX isa Matrix
    @test mX == hcat(X.x, X.y)
end
