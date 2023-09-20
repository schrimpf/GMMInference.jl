"""
    module GMMInference


"""
module GMMInference

using JuMP, ADNLPModels, Distributions, LinearAlgebra, Optim,
  ForwardDiff, ECOS
import Convex

export GMMModel,
  number_parameters,
  number_observations,
  number_moments,
  get_gi,
  gmm_objective,
  gmm_constraint,
  cue_objective,
  gel_nlp_problem,
  gel_jump_problem,
  gel_optim_args,
  gmm_nlp_problem,
  gmm_jump_problem,
  gel_pλ,
  gel_tests,
  IVLogitShare,
  MixturePanel,
  RCLogit

include("gmmmodel.jl")  # defines GMMModel and estimation methods
include("inference.jl") # test statistics and p-values

# example models
include("models/ivlogitshare.jl")
include("models/rclogit.jl")
include("models/mixturepanel.jl")
  
end # module
