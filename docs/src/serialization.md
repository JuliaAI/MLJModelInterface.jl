# Serialization

!!! warning "New in MLJBase 0.20"

	The following API is incompatible with versions of MLJBase < 0.20, even for model implementations compatible with MLJModelInterface 1^


This section may be occasionally relevant when wrapping models
implemented in languages other than Julia.

The MLJ user can serialize and deserialize machines, as she would any other julia
object. (This user has the option of first removing data from the machine. See the [Saving
machines](https://JuliaAI.github.io/MLJ.jl/dev/machines/#Saving-machines)
section of the MLJ manual for details.) However, a problem can occur if a model's
`fitresult` (see [The fit method](@ref)) is not a persistent object. For example, it might
be a C pointer that would have no meaning in a new Julia session.

If that is the case a model implementation needs to implement a `save`
and `restore` method for switching between a `fitresult` and some
persistent, serializable representation of that result.


## The save method

```julia
MMI.save(model::SomeModel, fitresult; kwargs...) -> serializable_fitresult
```

Implement this method to return a persistent serializable
representation of the `fitresult` component of the `MMI.fit` return
value.

The fallback of `save` performs no action and returns `fitresult`.


## The restore method

```julia
MMI.restore(model::SomeModel, serializable_fitresult) -> fitresult
```

Implement this method to reconstruct a valid `fitresult` (as would be returned by
`MMI.fit`) from a persistent representation constructed using
`MMI.save` as described above.

The fallback of `restore` performs no action and returns `serializable_fitresult`.


## Example

Refer to the model implementations at
[MLJXGBoostInterface.jl](https://github.com/JuliaAI/MLJXGBoostInterface.jl/blob/42afbd2974bd3bd734994004e367c98964ed1262/src/MLJXGBoostInterface.jl#L679).


