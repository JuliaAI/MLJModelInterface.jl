"""Cool model"""
@mlj_model mutable struct FooRegressor <: Deterministic
    a::Int = 0::(_ ≥ 0)
    b
end

struct BarGoo <: Deterministic end

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

@test_logs(
    (:warn, MLJModelInterface.WARN_MISSING_LOAD_PATH),
    metadata_model(BarGoo)
)

metadata_model(FooRegressor,
               input_scitype=Table(Continuous),
               target_scitype=AbstractVector{Continuous},
               supports_class_weights=true,
               load_path="goo goo")

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
    @test infos[:docstring] == "Cool model\n"
    @test infos[:name] == "FooRegressor"
    @test infos[:is_supervised]
    @test infos[:prediction_type] == :deterministic
    @test infos[:implemented_methods] == [:clean!]
    @test infos[:hyperparameters] == (:a, :b)
    @test infos[:hyperparameter_types] == ("Int64", "Any")
    @test infos[:hyperparameter_ranges] == (nothing, nothing)
end
