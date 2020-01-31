module MLJModelInterface

# ------------------------------------------------------------------------
# Dependency (note that ScientificTypes itself does not have dependencies)
using ScientificTypes
using InteractiveUtils # stdlib

# ------------------------------------------------------------------------
# exports

# types
export LightInterface, FullInterface
export MLJType, Model, Supervised, Unsupervised,
       Probabilistic, Deterministic, Interval, Static
# XXX export UnivariateFinite

# rexport types from ScientificTypes
export Scientific, Found, Unknown, Known, Finite, Infinite,
       OrderedFactor, Multiclass, Count, Continuous, Textual,
       Binary, ColorImage, GrayImage, Table

# model construction
export @mlj_model
# XXX metadata

# api
export fit, update, update_data, transform, inverse_transform,
       fitted_params, predict, predict_mode, predict_mean, predict_median,
       evaluate, clean!

# traits
export input_scitype, output_scitype, target_scitype,
       is_pure_julia, package_name, package_license,
       load_path, package_uuid, package_url,
       is_wrapper, supports_weights, supports_online,
       docstring, name, is_supervised,
       prediction_type, implemented_methods, hyperparameters,
       hyperparameter_types, hyperparameter_ranges

# data operations
export matrix, int, classes, decoder, table,
       nrows, selectrows, selectcols, select

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

include("utils.jl")

include("data_utils.jl")

include("model_traits.jl")
include("model_def.jl")
include("model_api.jl")

end # module
