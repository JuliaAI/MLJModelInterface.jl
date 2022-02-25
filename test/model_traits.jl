@mlj_model mutable struct S1 <: Supervised
end

@mlj_model mutable struct U1 <: Unsupervised
    a::Int
    b = sin
end

@mlj_model mutable struct D1 <: Deterministic
    a::Int = 1::(_ > 0)
end

@mlj_model mutable struct P1 <: Probabilistic
    a::Int = 1::(_ > 0)
end

@mlj_model mutable struct I1 <: Interval
end

@mlj_model mutable struct SA <: SupervisedAnnotator
end

@mlj_model mutable struct UA <: UnsupervisedAnnotator
end

foo(::P1) = 0
bar(::P1) = nothing

M.package_name(::Type{<:S1}) = "Sibelius"
M.package_url(::Type{<:S1}) = "www.find_the_eighth.org"
M.human_name(::Type{<:S1}) = "silly model"

M.package_name(::Type{<:U1}) = "Bach"
M.package_url(::Type{<:U1}) = "www.did_he_write_565.com"
M.human_name(::Type{<:U1}) = "my model"

@testset "traits" begin
    ms = S1()
    mu = U1(a=42, b=sin)
    md = D1()
    mp = P1()
    mi = I1()
    sa = SA()
    ua = UA()

    @test input_scitype(ms)  == Unknown
    @test output_scitype(ms) == Unknown
    @test target_scitype(ms) == Unknown
    @test is_pure_julia(ms)  == false

    @test package_name(ms) == "Sibelius"
    @test package_license(ms) == "unknown"
    @test load_path(ms) == "unknown"
    @test package_uuid(ms) == "unknown"
    @test package_url(ms) == "www.find_the_eighth.org"

    @test is_wrapper(ms) == false
    @test supports_online(ms) == false
    @test supports_weights(ms) == false
    @test iteration_parameter(ms) === nothing

    @test hyperparameter_ranges(md) == (nothing,)

    @test docstring(ms) == M.doc_header(S1)
    doc = docstring(mu) |> Markdown.parse
    comparison =
        """
        ```
        U1
        ```

        Model type for my model, based on
        [Bach.jl](www.did_he_write_565.com), and implementing the MLJ
        model interface.

        From MLJ, the type can be imported using

        ```
        U1 = @load U1 pkg=Bach
        ```

        Do `model = U1()` to construct an instance with default hyper-parameters.
        Provide keyword arguments to override hyper-parameter defaults, as in
        `U1(a=...)`.

        # Hyper-parameters

        - `a = 0`

        - `b = sin`

        """ |> Markdown.parse
    @test doc == comparison

    @test name(ms) == "S1"
    @test human_name(ms) == "silly model"


    @test is_supervised(ms)
    @test is_supervised(sa)
    @test !is_supervised(mu)
    @test !is_supervised(ua)
    @test prediction_type(ms) == :unknown
    @test prediction_type(md) == :deterministic
    @test prediction_type(mp) == :probabilistic
    @test prediction_type(mi) == :interval

    @test hyperparameters(md) == (:a,)
    @test hyperparameter_types(md) == ("Int64",)

    # implemented methods is deferred
    setlight()
    @test_throws M.InterfaceError implemented_methods(mp)

    setfull()

    function M.implemented_methods(::FI, M::Type{<:MLJType})
        return getfield.(methodswith(M), :name)
    end

    @test Set(implemented_methods(mp)) == Set([:clean!,:bar,:foo])
end

@testset "`_density` - helper for predict_scitype fallback" begin
    for T in [Continuous, Count, Textual]
        @test ==(
            M._density(AbstractArray{T,3}),
            AbstractArray{Density{T},3}
        )
    end

    for T in [Finite,
              Multiclass,
              OrderedFactor,
              Infinite,
              Continuous,
              Count,
              Textual]
        @test ==(
            M._density(AbstractVector{<:T}),
            AbstractVector{Density{<:T}}
        )
        @test M._density(Table(T)) == Table(Density{T})
    end

    for T in [Finite, Multiclass, OrderedFactor]
        @test ==(
            M._density(AbstractArray{<:T{2},3}),
            AbstractArray{Density{<:T{2}},3}
        )
        @test ==(
            M._density(AbstractArray{T{2},3}),
            AbstractArray{Density{T{2}},3}
        )
        @test M._density(Table(T{2})) == Table(Density{T{2}})
    end
end

@mlj_model mutable struct P2 <: Probabilistic end
M.target_scitype(::Type{<:P2}) = AbstractVector{<:Multiclass}
M.input_scitype(::Type{<:P2}) = Table(Continuous)

@mlj_model mutable struct U2 <: Unsupervised end
M.output_scitype(::Type{<:U2}) = AbstractVector{<:Multiclass}
M.input_scitype(::Type{<:U2}) = Table(Continuous)

@mlj_model mutable struct S2 <: Static end
M.output_scitype(::Type{<:S2}) = AbstractVector{<:Multiclass}
M.input_scitype(::Type{<:S2}) = Table(Continuous)

@testset "operation scitypes" begin
    @test predict_scitype(P2()) == AbstractVector{Density{<:Multiclass}}
    @test transform_scitype(P2()) == Unknown
    @test transform_scitype(U2()) == AbstractVector{<:Multiclass}
    @test inverse_transform_scitype(U2()) == Table(Continuous)
    @test predict_scitype(U2()) == Unknown
    @test transform_scitype(S2()) == AbstractVector{<:Multiclass}
    @test inverse_transform_scitype(S2()) == Table(Continuous)
end

@testset "abstract_type, fit_data_scitype" begin
    @test abstract_type(P2()) == Probabilistic
    @test abstract_type(S1()) == Supervised
    @test abstract_type(U1()) == Unsupervised
    @test abstract_type(D1()) == Deterministic
    @test abstract_type(P1()) == Probabilistic

    @test ==(
        fit_data_scitype(P2()),
        Tuple{Table(Continuous), AbstractVector{<:Multiclass}}
    )
    @test fit_data_scitype(U2()) == Tuple{Table(Continuous)}
    @test fit_data_scitype(S2()) == Tuple{}
end

true
