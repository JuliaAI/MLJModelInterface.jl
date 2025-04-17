module MLJModelInterface

const MODEL_TRAITS = [
    :input_scitype,
    :output_scitype,
    :target_scitype,
    :fit_data_scitype,
    :predict_scitype,
    :transform_scitype,
    :inverse_transform_scitype,
    :target_in_fit,
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
    :human_name,
    :is_supervised,
    :prediction_type,
    :abstract_type,
    :implemented_methods,
    :hyperparameters,
    :hyperparameter_types,
    :hyperparameter_ranges,
    :iteration_parameter,
    :supports_training_losses,
    :reports_feature_importances,
    :deep_properties,
    :reporting_operations,
    :constructor,
]

const ABSTRACT_MODEL_SUBTYPES = [
    :Supervised,
    :Unsupervised,
    :Probabilistic,
    :Deterministic,
    :Interval,
    :ProbabilisticSet,
    :JointProbabilistic,
    :Static,
    :Annotator,
    :SupervisedAnnotator,
    :UnsupervisedAnnotator,
    :SupervisedDetector,
    :UnsupervisedDetector,
    :ProbabilisticSupervisedDetector,
    :ProbabilisticUnsupervisedDetector,
    :DeterministicSupervisedDetector,
    :DeterministicUnsupervisedDetector
]


# ------------------------------------------------------------------------
# Dependencies
using ScientificTypesBase
using StatisticalTraits
using Random
using REPL # apparently needed to get Base.Docs.doc to work

import StatisticalTraits: info

# ------------------------------------------------------------------------
# exports

# mode
export LightInterface, FullInterface

# model types
export MLJType, Model

for T in ABSTRACT_MODEL_SUBTYPES
    @eval(export $T)
end

export UnivariateFinite

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
    nrows, selectrows, selectcols, select, info, scitype

# equality
export is_same_except, isrepresented

# re-exports from ScientificTypesBase
export Scientific, Found, Unknown, Known, Finite, Infinite,
    OrderedFactor, Multiclass, Count, Continuous, Textual,
    Binary, ColorImage, GrayImage, Image, Table, nonmissing

# ------------------------------------------------------------------------
# To be extended

import Base.==
import Base: in, isequal

# ------------------------------------------------------------------------
# Mode trick

struct LightInterface end
struct FullInterface end

const Mode = Union{LightInterface, FullInterface}

const INTERFACE_MODE = Ref{Mode}(LightInterface())

set_interface_mode(m::Mode) = (INTERFACE_MODE[] = m)

get_interface_mode() = INTERFACE_MODE[]

struct InterfaceError <: Exception
    m::String
end

abstract type MLJType end
abstract type Model <: MLJType end

# ------------------------------------------------------------------------
# Model subtypes

abstract type Supervised <: Model end
abstract type Unsupervised <: Model end
abstract type Annotator <: Model end

abstract type Probabilistic <: Supervised end
abstract type Deterministic <: Supervised end
abstract type Interval <: Supervised end
abstract type ProbabilisticSet <: Supervised end

abstract type JointProbabilistic <: Probabilistic end

abstract type Static <: Unsupervised end

abstract type SupervisedAnnotator <: Annotator end
abstract type UnsupervisedAnnotator <: Annotator end

abstract type UnsupervisedDetector <: UnsupervisedAnnotator end
abstract type SupervisedDetector <: SupervisedAnnotator end

abstract type ProbabilisticSupervisedDetector <: SupervisedDetector end
abstract type ProbabilisticUnsupervisedDetector <: UnsupervisedDetector end

abstract type DeterministicSupervisedDetector <: SupervisedDetector end
abstract type DeterministicUnsupervisedDetector <: UnsupervisedDetector end

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
