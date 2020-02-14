# MLJModelInterface.jl

A light-weight interface for developers wanting to integrate
machine learning models into
[MLJ](https://github.com/alan-turing-institute/MLJ.jl).


| [MacOS/Linux] | Coverage |
| :-----------: | :------: |
| [![Build Status](https://travis-ci.org/alan-turing-institute/MLJModelInterface.jl.svg?branch=master)](https://travis-ci.org/alan-turing-institute/MLJModelInterface.jl) | [![codecov.io](http://codecov.io/github/alan-turing-institute/MLJModelInterface.jl/coverage.svg?branch=master)](http://codecov.io/github/alan-turing-institute/MLJModelInterface.jl?branch=master) |


[MLJ](https://github.com/alan-turing-institute/MLJ.jl) is a framework
for evaluating, combining and optimizing machine learning models in
Julia. A third party package wanting to integrate their supervised or
unsupervised machine learning models must import the module
`MLJModelInterface` defined in this package. 

### Instructions

- [Quick-start guide](https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/) to adding models to MLJ

- [Detailed API
  specification](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/)
