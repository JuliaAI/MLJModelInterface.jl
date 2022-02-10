# poor man's info dict for testing
info_dict(MM::Type{<:Model}) =
    Dict(trait => eval(:($trait))(MM) for trait in M.MODEL_TRAITS)

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
metadata_model(FooRegressor,
    input=Table(Continuous),
    target=AbstractVector{Continuous},
    descr="La di da")

const HEADER = MLJModelInterface.doc_header(FooRegressor)

@doc """
$HEADER

Yes, we have no bananas. We have no bananas today!
""" FooRegressor

@testset "metadata" begin
    setfull()
    M.implemented_methods(::FI, M::Type{<:MLJType}) =
        getfield.(methodswith(M), :name)
    infos = info_dict(FooRegressor)

    @test infos[:input_scitype] == Table(Continuous)
    @test infos[:output_scitype] == Unknown
    @test infos[:target_scitype] == AbstractVector{Continuous}
    @test infos[:is_pure_julia]
    @test infos[:package_name] == "FooRegressorPkg"
    @test infos[:package_license] == "MIT"
    @test infos[:load_path] == "MLJModels.FooRegressorPkg_.FooRegressor"
    @test infos[:package_uuid] == "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    @test infos[:package_url] == "http://existentialcomics.com/"
    @test !infos[:is_wrapper]
    @test !infos[:supports_weights]
    @test !infos[:supports_online]
    @test startswith(infos[:docstring], "La di da")
    @test infos[:name] == "FooRegressor"
    @test infos[:is_supervised]
    @test infos[:prediction_type] == :deterministic
    @test infos[:implemented_methods] == [:clean!]
    @test infos[:hyperparameters] == (:a, :b)
    @test infos[:hyperparameter_types] == ("Int64", "Any")
    @test infos[:hyperparameter_ranges] == (nothing, nothing)
end

@testset "doc_header(ModelType)" begin

    # we test markdown parsed strings for less fussy comparison

    header  = Markdown.parse(HEADER)
    comparison =
"""
```
FooRegressor
```

Model type for foo regressor, based on [FooRegressorPkg.jl](http://existentialcomics.com/).

From MLJ, the type can be imported using

```
FooRegressor = @load FooRegressor pkg=FooRegressorPkg
```

Do `model = FooRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter
defaults, as in `FooRegressor(a=...)`.
""" |> chomp |> Markdown.parse

end

@testset "document string" begin
    doc = (@doc FooRegressor) |> string |> chomp
    @test endswith(doc, "We have no bananas today!")
end
