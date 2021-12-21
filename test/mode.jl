@testset "light" begin
    M.set_interface_mode(M.LightInterface())
    @test M.get_interface_mode() isa M.LightInterface

    # For matrix objects (:other) we don't need `FullInterface`
    # to run `matrix` method as they are already matrices.
    X   = zeros(3, 4)
    mX  = matrix(X)
    mtX = matrix(X; transpose=true)

    @test mX === X
    @test mtX == permutedims(X)

    # for other objects `:table` or `:other` we need
    # `FullInterface` to get the corresponding trait
    X = (x=[1, 2, 3], y=[1, 2, 3])
    @test_throws M.InterfaceError M.vtrait(X)
    @test_throws M.InterfaceError matrix(X)
    X = (1, 2, 3, 4)
    @test_throws M.InterfaceError M.vtrait(X)
    @test_throws M.InterfaceError matrix(X)
end

@testset "full" begin
    M.set_interface_mode(M.FullInterface())
    @test M.get_interface_mode() isa M.FullInterface

    M.matrix(::M.FullInterface, ::Val{:table}, X; kw...) = Tables.matrix(X; kw...)
    M.vtrait(::FI, X, s) = Val{ifelse(Tables.istable(X), :table, :other)}()
    
    X = (x=[1, 2, 3], y=[1, 2, 3])
    mX = matrix(X)
    @test mX isa Matrix
    @test mX == hcat(X.x, X.y)
end
