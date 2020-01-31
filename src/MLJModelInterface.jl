module MLJModelInterface

# ------------------------------------------------------------------------
# Dependency (note that ScientificTypes itself does not have dependencies)
import ScientificTypes: trait

# ------------------------------------------------------------------------
# exports

# types
export LightInterface, FullInterface
export MLJType, Model, Supervised, Unsupervised,
       Probabilistic, Deterministic, Interval, Static
# export UnivariateFinite

# model construction
export @mlj_model

# operations
# export fit, predict, ...
export matrix, int, classes, decoder, table,
       nrows, select, selectrows, selectcols

# ------------------------------------------------------------------------
# Mode trick

abstract type Mode end
struct LightInterface <: Mode end
struct FullInterface  <: Mode end

const INTERFACE_MODE = Ref{Mode}(LightInterface())

set_interface_mode(m::Mode) = (INTERFACE_MODE[] = m)

get_interface_mode() = INTERFACE_MODE[]

struct InterfaceError <: Exception
    m::String
end

# ------------------------------------------------------------------------
# Model types

abstract type MLJType end

abstract type Model <: MLJType end

abstract type   Supervised <: Model end
abstract type Unsupervised <: Model end

abstract type Probabilistic <: Supervised end
abstract type Deterministic <: Supervised end
abstract type      Interval <: Supervised end

abstract type Static <: Unsupervised end

# ------------------------------------------------------------------------
# includes

include("data_utils.jl")

include("mlj_model.jl")

end # module
