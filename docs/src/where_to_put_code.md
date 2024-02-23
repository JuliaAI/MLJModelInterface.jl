# Where to place code implementing new models

Note that different packages can implement models having the same name
without causing conflicts, although an MLJ user cannot simultaneously
*load* two such models.

There are two options for making a new model implementation available
to all MLJ users:

1. *Native implementations* (preferred option). The implementation
   code lives in the same package that contains the learning
   algorithms implementing the interface. An example is
   [`EvoTrees.jl`](https://github.com/Evovest/EvoTrees.jl/blob/master/src/MLJ.jl). In
   this case, it is sufficient to open an issue at
   [MLJ](https://github.com/alan-turing-institute/MLJ.jl) requesting
   the package to be registered with MLJ. Registering a package allows
   the MLJ user to access its models' metadata and to selectively load
   them.

2. *Separate interface package*. Implementation code lives in a
   separate *interface package*, which has the algorithm-providing
   package as a dependency. See the template repository
   [MLJExampleInterface.jl](https://github.com/JuliaAI/MLJExampleInterface.jl).

Additionally, one needs to ensure that the implementation code defines
the `package_name` and `load_path` model traits appropriately, so that
`MLJ`'s `@load` macro can find the necessary code (see
[MLJModels/src](https://github.com/JuliaAI/MLJModels.jl/tree/master/src)
for examples).
