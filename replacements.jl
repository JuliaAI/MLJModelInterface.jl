abstract type          Supervised <: Model end
abstract type        Unsupervised <: Model end
abstract type           Annotator <: Model end

abstract type Probabilistic <: Supervised end
abstract type Deterministic <: Supervised end
abstract type      Interval <: Supervised end

abstract type JointProbabilistic <: Probabilistic end

abstract type Static                <: Unsupervised end

abstract type SupervisedAnnotator   <: Annotator end
abstract type UnsupervisedAnnotator <: Annotator end

abstract type UnsupervisedDetector <: UnsupervisedAnnotator end
abstract type SupervisedDetector   <: SupervisedAnnotator end

abstract type ProbabilisticSupervisedDetector   <: SupervisedDetector end
abstract type ProbabilisticUnsupervisedDetector <: UnsupervisedDetector end

abstract type DeterministicSupervisedDetector   <: SupervisedDetector end
abstract type DeterministicUnsupervisedDetector <: UnsupervisedDetector end

const ABSTRACT_MODEL_SUBTYPES =
    [:Supervised,
     :Unsupervised,
     :Probabilistic,
     :Deterministic,
     :Interval,
     :JointProbabilistic,
     :Static,
     :Annotator,
     :SupervisedAnnotator,
     :UnsupervisedAnnotator,
     :SupervisedDetector,
     :UnsupervisedDetector,
     :ProbabilisticSupervisedDetector,
     :ProbabilisticUnsupervisedDetector,
     :DeterministicSupervisedDetector,
     :DeterministicUnsupervisedDetector]

MLJBase.Unsupervised -> Model