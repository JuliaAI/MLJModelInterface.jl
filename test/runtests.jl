using Test, MLJModelInterface, ScientificTypes
using Tables, Distances, CategoricalArrays

const M  = MLJModelInterface
const FI = M.FullInterface
const CategoricalElement = Union{CategoricalValue,CategoricalString}
ScientificTypes.TRAIT_FUNCTION_GIVEN_NAME[:table] = Tables.istable

setlight() = M.set_interface_mode(M.LightInterface())
setfull()  = M.set_interface_mode(M.FullInterface())

include("mode.jl")
include("data_utils.jl")

include("mlj_model.jl")
