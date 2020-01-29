module MLJModelInterface

# ------------------------------------------------------------------------
# Dependency (note that ScientificTypes itself does not have dependencies)
import ScientificTypes: trait

# ------------------------------------------------------------------------
# Single export: matrix, everything else is qualified in MLJBase
export matrix

# ------------------------------------------------------------------------

abstract type Mode end
struct LightInterface <: Mode end
struct FullInterface  <: Mode end

const INTERFACE_MODE = Ref{Mode}(LightInterface())

set_interface_mode(m::Mode) = (INTERFACE_MODE[] = m)

get_interface_mode() = INTERFACE_MODE[]

struct InterfaceError <: Exception
    m::String
end

vtrait(X) = X |> trait |> Val

"""
    matrix(X; transpose=false)

If `X <: AbstractMatrix`, return `X` or `permutedims(X)` if `transpose=true`.
If `X` is a Tables.jl compatible table source, convert `X` into a `Matrix`.
"""
matrix(X; kw...) = matrix(vtrait(X), X, get_interface_mode(); kw...)

matrix(::Val{:other}, X::AbstractMatrix, ::Mode; transpose=false) =
    transpose ? permutedims(X) : X

matrix(::Val{:other}, X, ::Mode; kw...) =
    throw(ArgumentError("Function `matrix` only supports AbstractMatrix or " *
                        "containers implementing the Tables interface."))

matrix(::Val{:table}, X, ::LightInterface; kw...) =
    throw(InterfaceError("Only `MLJModelInterface` loaded. Import `MLJBase`."))

end # module
