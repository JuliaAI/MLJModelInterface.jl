using Test, MLJModelInterface
using ScientificTypesBase, ScientificTypes
using Tables, Distances, CategoricalArrays, InteractiveUtils
import DataFrames: DataFrame

const M  = MLJModelInterface
const FI = M.FullInterface

setlight() = M.set_interface_mode(M.LightInterface())
setfull() = M.set_interface_mode(M.FullInterface())

include("mode.jl")
include("parameter_inspection.jl")
include("data_utils.jl")
include("metadata_utils.jl")
include("model_def.jl")
include("model_api.jl")
include("model_traits.jl")
include("equality.jl")
