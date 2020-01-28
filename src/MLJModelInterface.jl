module MLJModelInterface

# ------------------------------------------------------------------------
# Dependency (note that ScientificTypes itself does not have dependencies)
using ScientificTypes

# ------------------------------------------------------------------------
# Exports
export Dummy, Live, get_interface_mode
export matrix

# ------------------------------------------------------------------------
# Mode trick

abstract type Mode end
struct Dummy <: Mode end
struct Live  <: Mode end

const INTERFACE_MODE = Ref{Mode}(Dummy())

get_interface_mode() = INTERFACE_MODE[]

matrix(a...; kw...) = matrix(a...; interface_mode=get_interface_mode(), kw...)

matrix(a...; interface_mode::Mode=Dummy(), kw...) =
    error("Only `MLJModelInterface` loaded. Do `import MLJBase`.")

end # module
