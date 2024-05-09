# Document strings

To be registered, MLJ models must include a detailed document string
for the model type, and this must conform to the standard outlined
below. We recommend you simply adapt an existing compliant document
string and read the requirements below if you're not sure, or to use
as a checklist. Here are examples of compliant doc-strings (go to the
end of the linked files):

- Regular supervised models (classifiers and regressors): [MLJDecisionTreeInterface.jl](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl/blob/master/src/MLJDecisionTreeInterface.jl) (see the end of the file)

- Tranformers: [MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl/blob/dev/src/builtins/Transformers.jl)

A utility function is available for generating a standardized header
for your doc-strings (but you provide most detail by hand):

```@docs
MLJModelInterface.doc_header
```

## The document string standard

Your document string must include the following components, in order:

- A *header*, closely matching the example given above.

- A *reference describing the algorithm* or an actual description of
  the algorithm, if necessary. Detail any non-standard aspects of the
  implementation. Generally, defer details on the role of
  hyperparameters to the "Hyperparameters" section (see below).

- Instructions on *how to import the model type* from MLJ (because a user can
  already inspect the doc-string in the Model Registry, without having loaded
  the code-providing package).

- Instructions on *how to instantiate* with default hyperparameters or with keywords.

- A *Training data* section: explains how to bind a model to data in a machine
  with all possible signatures (eg, `machine(model, X, y)` but also
  `machine(model, X, y, w)` if, say, weights are supported);  the role and
  scitype requirements for each data argument should be itemized.

- Instructions on *how to fit* the machine (in the same section).

- A *Hyperparameters* section (unless there aren't any): an itemized list of the parameters, with defaults given.

- An *Operations* section: each implemented operation (`predict`,
  `predict_mode`, `transform`, `inverse_transform`, etc ) is itemized and
  explained. This should include operations with no data arguments, such as
  `training_losses` and `feature_importances`.

- A *Fitted parameters* section: To explain what is returned by `fitted_params(mach)`
  (the same as `MLJModelInterface.fitted_params(model, fitresult)` -  see later)
  with the fields of that named tuple itemized.

- A *Report* section (if `report` is non-empty): To explain what, if anything,
  is included in the `report(mach)`  (the same as the `report` return value of
  `MLJModelInterface.fit`) with the fields itemized.

- An optional but highly recommended *Examples* section, which includes MLJ
  examples, but which could also include others if the model type also
  implements a second "local" interface, i.e., defined in the same module. (Note
  that each module referring to a type can declare separate doc-strings which
  appear concatenated in doc-string queries.)

- A closing *"See also"* sentence which includes a `@ref` link to the raw model type (if you are wrapping one).


