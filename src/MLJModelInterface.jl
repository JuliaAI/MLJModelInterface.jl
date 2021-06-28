module MLJModelInterface

const MODEL_TRAITS = [
    :input_scitype,
    :output_scitype,
    :target_scitype,
    :is_pure_julia,
    :package_name,
    :package_license,
    :load_path,
    :package_uuid,
    :package_url,
    :is_wrapper,
    :supports_weights,
    :supports_class_weights,
    :supports_online,
    :docstring,
    :name,
    :is_supervised,
    :prediction_type,
    :implemented_methods,
    :hyperparameters,
    :hyperparameter_types,
    :hyperparameter_ranges,
    :iteration_parameter,
    :supports_training_losses,
    :deep_properties]

# ------------------------------------------------------------------------
# Dependencies
using ScientificTypesBase
using StatisticalTraits
using Random

# ------------------------------------------------------------------------
# exports

# mode
export LightInterface, FullInterface

# MLJ model hierarchy
export MLJType, Model, Supervised, Unsupervised,
       Probabilistic, JointProbabilistic, Deterministic, Interval, Static,
       UnivariateFinite

# parameter_inspection:
export params

# model constructor + metadata
export @mlj_model, metadata_pkg, metadata_model

# model api
export fit, update, update_data, transform, inverse_transform,
    fitted_params, predict, predict_mode, predict_mean, predict_median,
    predict_joint, evaluate, clean!, reformat, training_losses

# model traits
for trait in MODEL_TRAITS
    @eval(export $trait)
end

# data operations
export matrix, int, classes, decoder, table,
       nrows, selectrows, selectcols, select

# equality
export is_same_except, isrepresented

# re-exports from ScientificTypesBase
export Scientific, Found, Unknown, Known, Finite, Infinite,
       OrderedFactor, Multiclass, Count, Continuous, Textual,
       Binary, ColorImage, GrayImage, Image, Table
export scitype, scitype_union, elscitype, nonmissing, trait, info

# ------------------------------------------------------------------------
# To be extended

import Base.==
import Base: in, isequal
#
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

abstract type Model   <: MLJType end

abstract type   Supervised <: Model end
abstract type Unsupervised <: Model end

abstract type Probabilistic <: Supervised end
abstract type Deterministic <: Supervised end
abstract type      Interval <: Supervised end

abstract type Static <: Unsupervised end

abstract type JointProbabilistic <: Probabilistic end

# ------------------------------------------------------------------------
# includes

include("parameter_inspection.jl")
include("data_utils.jl")
include("metadata_utils.jl")
include("model_traits.jl")
include("model_def.jl")
include("model_api.jl")
include("equality.jl")


end # module
