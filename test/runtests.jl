using Test, MLJModelInterface
using ScientificTypesBase, ScientificTypes
using Tables, Distances, CategoricalArrays, InteractiveUtils
import DataFrames: DataFrame
import Markdown
import OrderedCollections
import Aqua

const M  = MLJModelInterface
const FI = M.FullInterface

setlight() = M.set_interface_mode(M.LightInterface())
setfull() = M.set_interface_mode(M.FullInterface())

@testset "mode.jl" begin
    include("mode.jl")
end

@testset "parameter_inspection.jl" begin
    include("parameter_inspection.jl")
end

@testset "data_utils.jl" begin
    include("data_utils.jl")
end

@testset "metadata_utils.jl" begin
    include("metadata_utils.jl")
end
@testset "model_def.jl" begin
    include("model_def.jl")
end

@testset "model_api.jl" begin
    include("model_api.jl")
end

@testset "model_traits.jl" begin
    include("model_traits.jl")
end

@testset "equality.jl" begin
    include("equality.jl")
end

# @testset "aqua.jl" begin
#     include("aqua.jl")
# end
