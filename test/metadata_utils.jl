##  SYNTHETIC DOCSTRING

# model type `M` for which `M()` is implemented (typical case):

"""Cool model"""
@mlj_model mutable struct FooClassifier <: Deterministic
    a::Int = 0::(_ ≥ 0)
    b
end

metadata_pkg(FooClassifier,
    name="FooClassifierPkg",
    uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2",
    url="http://existentialcomics.com/",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

metadata_model(FooClassifier,
               input_scitype=Table(Continuous),
               target_scitype=AbstractVector{Continuous},
               supports_class_weights=true,
               load_path="goo goo")


# model type `M` for which `M()` is not implemented:

"""Cool model"""
mutable struct FooBad <: Deterministic
    a::Int
    b
end

metadata_pkg(FooBad,
    name="FooBadPkg",
    uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2",
    url="http://existentialcomics.com/",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

metadata_model(FooBad,
               input_scitype=Table(Continuous),
               target_scitype=AbstractVector{Continuous},
               supports_class_weights=true,
               load_path="goo goo")

synthetic = M.synthesize_docstring(FooClassifier) |> Markdown.parse
comparison =
    """
        FooClassifier

    A model type for constructing a foo classifier, based on
    [FooClassifierPkg.jl](http://existentialcomics.com/),
    and implementing the MLJ model interface.

    From MLJ, the type can be imported using

        FooClassifier = @load FooClassifier pkg=FooClassifierPkg

    Do `model = FooClassifier()` to construct an instance with default hyper-parameters.
    Provide keyword arguments to override hyper-parameter defaults, as in
    `FooClassifier(a=...)`.

    # Hyper-parameters

    - `a = 0`

    - `b = missing`
    """ |> Markdown.parse

@testset "synthisise_docstring" begin
    @test synthetic == comparison
    @test isempty(M.synthesize_docstring(FooBad))
end


# METADATA_PKG, METADATA_MODEL, DOCUMENT STRINGS


"""Cool model"""
@mlj_model mutable struct FooRegressor <: Deterministic
    a::Int = 0::(_ ≥ 0)
    b
end

metadata_pkg(FooRegressor,
    name="FooRegressorPkg",
    uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2",
    url="http://existentialcomics.com/",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

# this is added in MLJBase but not in MLJModelInterface, to avoid
# InteractiveUtils as dependency:
setfull()
M.implemented_methods(::FI, M::Type{<:MLJType}) =
    getfield.(methodswith(M), :name)

metadata_model(FooRegressor,
               input_scitype=Table(Continuous),
               target_scitype=AbstractVector{Continuous},
               supports_class_weights=true,
               load_path="goo goo")

const HEADER = MLJModelInterface.doc_header(FooRegressor)

@doc """
$HEADER

Yes, we have no bananas. We have no bananas today!
""" FooRegressor


infos =  Dict(trait => eval(:(MLJModelInterface.$trait))(FooRegressor) for
              trait in M.MODEL_TRAITS)

@testset "metadata" begin
    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:output_scitype] == Unknown
    @test infos[:target_scitype] == AbstractVector{Continuous}
    @test infos[:is_pure_julia]
    @test infos[:package_name] == "FooRegressorPkg"
    @test infos[:package_license] == "MIT"
    @test infos[:load_path] == "goo goo"
    @test infos[:package_uuid] == "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    @test infos[:package_url] == "http://existentialcomics.com/"
    @test !infos[:is_wrapper]
    @test !infos[:supports_weights]
    @test infos[:supports_class_weights]
    @test !infos[:supports_online]
    @test infos[:docstring] == (@doc FooRegressor) |> string
    @test infos[:name] == "FooRegressor"
    @test infos[:human_name] == "foo regressor"
    @test infos[:is_supervised]
    @test infos[:prediction_type] == :deterministic
    @test infos[:implemented_methods] == [:clean!]
    @test infos[:hyperparameters] == (:a, :b)
    @test infos[:hyperparameter_types] == ("Int64", "Any")
    @test infos[:hyperparameter_ranges] == (nothing, nothing)
    @test !infos[:supports_training_losses] 
    @test !infos[:reports_feature_importances]
end

@testset "doc_header(ModelType)" begin

    # we test markdown parsed strings for less fussy comparison

    header  = Markdown.parse(HEADER)
    comparison =
"""
```
FooRegressor
```

A model type for constructing a foo regressor, based on
[FooRegressorPkg.jl](http://existentialcomics.com/), and implementing the MLJ model
interface.

From MLJ, the type can be imported using

```
FooRegressor = @load FooRegressor pkg=FooRegressorPkg
```

Do `model = FooRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter
defaults, as in `FooRegressor(a=...)`.
""" |> chomp |> Markdown.parse

    @test string(header) == string(comparison)
end

@testset "document string" begin
    doc = (@doc FooRegressor) |> string |> chomp
    @test endswith(doc, "We have no bananas today!")
end


# # DOC STRING - AUGMENTED CASE

"""Cool model"""
@mlj_model mutable struct FooRegressor2 <: Deterministic
    a::Int = 0::(_ ≥ 0)
    b
end

metadata_pkg(FooRegressor2,
    name="FooRegressor2Pkg",
    uuid="10745b16-79ce-11e8-11f9-7d13ad32a3b2",
    url="http://existentialcomics.com/",
    julia=true,
    license="MIT",
    is_wrapper=false
    )

const HEADER2 = MLJModelInterface.doc_header(FooRegressor2, augment=true)

@doc """
$HEADER2

Yes, we have no bananas. We have no bananas today!
""" FooRegressor2

@testset "doc_header(ModelType, augment=true)" begin

    # we test markdown parsed strings for less fussy comparison

    header  = Markdown.parse(HEADER2)
    comparison =
"""
From MLJ, the `FooRegressor2` type can be imported using

```
FooRegressor2 = @load FooRegressor2 pkg=FooRegressor2Pkg
```

Do `model = FooRegressor2()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter
defaults, as in `FooRegressor2(a=...)`.
""" |> chomp |> Markdown.parse

    @test string(header) == string(comparison)

end

@testset "document string" begin
    doc = (@doc FooRegressor2) |> string |> chomp
    @test endswith(doc, "We have no bananas today!")
end
