# MLJModelInterface.jl

A light-weight interface for developers wanting to integrate
machine learning models into
[MLJ](https://github.com/JuliaAI/MLJ.jl).


| Linux | Coverage |
| :-----------: | :------: |
| [![Build Status](https://github.com/JuliaAI/MLJModelInterface.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJModelInterface.jl/actions) | [![codecov](https://codecov.io/github/JuliaAI/MLJModelInterface.jl/graph/badge.svg?token=rkvwHku1dW)](https://codecov.io/github/JuliaAI/MLJModelInterface.jl)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaai.github.io/MLJModelInterface.jl/dev/)


[MLJ](https://JuliaAI.github.io/MLJ.jl/dev/) is a framework for evaluating,
combining and optimizing machine learning models in Julia. A third party package wanting
to integrate their machine learning models into MLJ must import the module
`MLJModelInterface` defined in this package, as described in the
[documentation](https://JuliaAI.github.io/MLJModelInterface.jl/dev/).
