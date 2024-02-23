using Documenter
using MLJModelInterface
import MLJModelInterface as MMI

makedocs(;
         modules=[MLJModelInterface, ],
         format=Documenter.HTML(),
         pages=[
             "Home" => "index.md",
             "Quick-start guide" => "quick_start_guide.md",
             "The model type hierarchy" => "the_model_type_hierarchy.md",
             "New model type declarations" => "type_declarations.md",
             "Supervised models" => "supervised_models.md",
             "Summary of methods" => "summary_of_methods.md",
             "The form of data for fitting and predicting" => "form_of_data.md",
             "The fit method" => "the_fit_method.md",
             "The fitted_params method" => "the_fitted_params_method.md",
             "The predict method" => "the_predict_method.md",
             "The predict_joint method" => "the_predict_joint_method.md",
             "Training losses" => "training_losses.md",
             "Feature importances" =>  "feature_importances.md",
             "Trait declarations" => "trait_declarations.md",
             "Iterative models and the update! method" => "iterative_models.md",
             "Implementing a data front end" => "implementing_a_data_front_end.md",
             "Supervised models with a transform method" =>
                 "supervised_models_with_transform.md",
             "Models that learn a probability distribution" => "fitting_distributions.md",
             "Serialization" => "serialization.md",
             "Document strings" => "document_strings.md",
             "Unsupervised models" => "unsupervised_models.md",
             "Static models" => "static_models.md",
             "Outlier detection models" => "outlier_detection_models.md",
             "Convenience methods" => "convenience_methods.md",
             "Where to place code implementing new models" => "where_to_put_code.md",
             "How to add models to the MLJ Model Registry" => "how_to_register.md",
             "Reference" => "reference.md",
         ],
         sitename="MLJModelInterface",
         warnonly = [:cross_references, :missing_docs],
)

deploydocs(
    repo = "github.com/JuliaAI/MLJModelInterface.jl",
    devbranch="dev",
    push_preview=false,
)
