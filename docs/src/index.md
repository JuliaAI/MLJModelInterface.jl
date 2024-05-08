# Adding Models for General Use

The machine learning tools provided by
[MLJ](https://JuliaAI.github.io/MLJ.jl/dev/) can be applied to the models in
any package that imports 
[MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl) and implements the
API defined there, as outlined in this document. 

!!! tip

    This is a reference document, which has become rather sprawling over the evolution of the MLJ project. We recommend starting with [Quick start guide](@ref), which covers the main points relevant to most new model implementations.

Interface code can be hosted by the package providing the core machine learning algorithm,
or by a stand-alone "interface-only" package, using the template
[MLJExampleInterface.jl](https://github.com/JuliaAI/MLJExampleInterface.jl) (see [Where to
place code implementing new models](@ref) below). For a list of packages implementing the
MLJ model API (natively, and in interface packages) see
[here](https://JuliaAI.github.io/MLJ.jl/dev/list_of_supported_models/).

## Important

[MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl)
is a very light-weight interface allowing you to *define* your
interface, but does not provide the functionality required to use or
test your interface; this requires
[MLJBase](https://github.com/JuliaAI/MLJBase.jl).  So,
while you only need to add `MLJModelInterface` to your project's
[deps], for testing purposes you need to add
[MLJBase](https://github.com/JuliaAI/MLJBase.jl) to your
project's [extras] and [targets]. In testing, simply use `MLJBase` in
place of `MLJModelInterface`.

It is assumed the reader has read the [Getting
Started](https://JuliaAI.github.io/MLJ.jl/dev/getting_started/) section of
the MLJ manual.  To implement the API described here, some familiarity with the following
packages is also helpful:

- [ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl)
  (for specifying model requirements of data)

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
  (for probabilistic predictions)

- [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl)
  (essential if you are implementing a model handling data of
  `Multiclass` or `OrderedFactor` scitype; familiarity with
  `CategoricalPool` objects required)

- [Tables.jl](https://github.com/JuliaData/Tables.jl) (if your
  algorithm needs input data in a novel format).

In MLJ, the basic interface exposed to the user, built atop the model interface described
here, is the *machine interface*. After a first reading of this document, the reader may
wish to refer to [MLJ
Internals](https://JuliaAI.github.io/MLJ.jl/dev/internals/) for context.

