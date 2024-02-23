# The form of data for fitting and predicting

The model implementer does not have absolute control over the types of
data `X`, `y` and `Xnew` appearing in the `fit` and `predict` methods
they must implement. Rather, they can specify the *scientific type* of
this data by making appropriate declarations of the traits
`input_scitype` and `target_scitype` discussed later under [Trait
declarations](@ref).

*Important Note.* Unless it genuinely makes little sense to do so, the
MLJ recommendation is to specify a `Table` scientific type for `X`
(and hence `Xnew`) and an `AbstractVector` scientific type (e.g.,
`AbstractVector{Continuous}`) for targets `y`. Algorithms requiring
matrix input can coerce their inputs appropriately; see below.


## Additional type coercions

If the core algorithm being wrapped requires data in a different or
more specific form, then `fit` will need to coerce the table into the
form desired (and the same coercions applied to `X` will have to be
repeated for `Xnew` in `predict`). To assist with common cases, MLJ
provides the convenience method
[`MMI.matrix`](@ref). `MMI.matrix(Xtable)` has type `Matrix{T}` where
`T` is the tightest common type of elements of `Xtable`, and `Xtable`
is any table. (If `Xtable` is itself just a wrapped matrix,
`Xtable=Tables.table(A)`, then `A=MMI.table(Xtable)` will be returned
without any copying.)

Alternatively, a more performant option is to implement a data
front-end for your model; see [Implementing a data front-end](@ref).

Other auxiliary methods provided by MLJModelInterface for handling tabular data
are: `selectrows`, `selectcols`, `select` and `schema` (for extracting
the size, names and eltypes of a table's columns). See [Convenience
methods](@ref) below for details.


## Important convention

It is to be understood that the columns of table `X` correspond to
features and the rows to observations. So, for example, the predict
method for a linear regression model might look like `predict(model,
w, Xnew) = MMI.matrix(Xnew)*w`, where `w` is the vector of learned
coefficients.


