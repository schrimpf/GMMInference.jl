"""
    GMMModel

Abstract type for GMM models.
"""
abstract type GMMModel end

"""
    get_gi(model::GMMModel)

Returns a function `gi(θ)` where the moment condition for a GMM model is

``E[g_i(\\theta)] = 0``

`gi(θ)` should be a number of observations by number of moment conditions matrix. 
"""
function get_gi(model::GMMModel)
  error("get_gi not defined for model type $(typeof(model))")
end

""" 
     number_parameters(model::GMMModel)

Number of parameters (dimension of θ) for a GMMModel.
"""
function number_parameters(model::GMMModel)
  error("number_parameters not defined for model type $(typeof(model))")
end

"""
     number_observations(model::GMMModel)

Number of observations (rows of gi(θ)) for a GMMModel
"""
function number_observations(model::GMMModel)  
  θ = ones(number_parameters(model))
  gi = get_gi(model)
  return(size(gi(θ),1))
end

"""
    number_moments(model::GMModel)

Number of moments (columns of gi(θ)) for a GMMModel
"""
function number_moments(model::GMMModel)
  θ = ones(number_parameters(model))
  gi = get_gi(model)
  return(size(gi(θ),2))
end


""" 
    gmm_constraints(model::GMMModel)

Returns constraint as function or parameters, `θ`, where `c(θ) = 0`.
"""
function gmm_constraints(model::GMMModel)
  return nothing
end

"""
     gmm_objective(gi::Function, W=I)

Returns ``Q(θ) = n (1/n \\sum_i g_i(θ)) W (1/n \\sum_i g_i(θ))' ``
"""
function gmm_objective(gi::Function, W=I)
  function(θ)
    g = gi(θ)
    m = mean(g,dims=1)
    (size(g)[1]*( m*W*m')[1])
  end
end

"""
     cue_objective(gi::Function)

Returns the CUE objective function for moment functions gi. 

``Q(θ) = n (1/n \\sum_i g_i(θ)) \\widehat{cov}(g_i(θ))^{-1} (1/n \\sum_i g_i(θ))' ``

Calculates cov(gi(θ)) assuming observations are independent.
"""
function cue_objective(gi::Function)
  function(θ)
    g = gi(θ)
    m = mean(g,dims=1)
    W = inv(cov(g))
    (size(g)[1]*( m*W*m')[1])
  end
end


"""  
    gmm_objective(model::GMMModel, W=I)

Returns the GMM objective function with weighting matrix `W` for `model`.
"""
function gmm_objective(model::GMMModel, W=I)
  gmm_objective(get_gi(model), W)
end

"""  
    cue_objective(model::GMMModel)

Returns the CUE objective function for `model`. 

Calculated weighting matrix assuming observations are independent.
"""
function cue_objective(model::GMMModel)
  cue_objective(get_gi(model))
end

"""
    gel_jump_problem(model::GMMModel, h::Function=log)

Returns JuMP problem for generalized empirical likelihood estimation of `model`. 
`h` is a generalized likelihood function. 
"""
function gel_jump_problem(model::GMMModel, h::Function=log)
  error("gel_jump_problem not defined for model type $(typeof(model))")
end

"""
    gel_nlp_problem(model::GMMModel, h::Function=log)

Returns NLPModel problem for generalized empirical likelihood estimation of `model`. 
`h` is a generalized likelihood function. 
"""
function gel_nlp_problem(model::GMMModel, h::Function=log)
  gi = get_gi(model)
  n = number_observations(model)
  d = number_parameters(model)
  k = number_moments(model)
  el(x) = -sum(h.(x[1:n]))
  c = gmm_constraints(model)
  nc = isnothing(c) ? 0 : length(c(zeros(d)))
  con = function(x)
    p = x[1:n]
    θ = x[(n+1):end]
    if isnothing(c)      
      out=[sum(p) sum(gi(θ).*p, dims=1)]
    else
      out=[sum(p) sum(gi(θ).*p, dims=1) c(θ)]
    end
    return(vec(out))
  end
  lcon = [1.0, zeros(k+nc)...]
  ucon = [1.0, zeros(k+nc)...]
  lvar = [zeros(n)... , fill(-1e10,d)...]
  uvar = [ones(n)... ,  fill(1e10,d)...]
  name = "GEL for $(typeof(model))"
  x0 = [fill(1/n,n)... , ones(d)...]
  return(ADNLPModel(el,x0, lvar, uvar, 
                    con, lcon, ucon,
                    name=name))
end

"""
    gmm_nlp_problem(model::GMMModel, obj=gmm_objective)

Constructs NLPModel for GMMModel. 
"""
function gmm_nlp_problem(model::GMMModel, obj=gmm_objective)
  con = gmm_constraints(model)
  if con===nothing
    return(ADNLPModel(obj(model), ones(number_parameters(model)),
                      name="$(typeof(model))"))
  else
    θ = zeros(number_parameters(model))
    nc = length(con(θ))
    return(ADNLPModel(obj(model), ones(number_parameters(model)),
                      c=con, lcon=zeros(nc), ucon=zeros(nc),
                      name="$(typeof(model))"))  
  end
end

"""
    gmm_jump_problem(model::GMMModel, obj=gmm_objective)

Constructs JuMP problem for GMMModel. 
"""
function gmm_jump_problem(model::GMMModel, obj=gmm_objective)
  con = gmm_constraints(model)
  m = Model()
  d = number_parameters(model)
  @variable(m, θ[1:d])
  if con!==nothing
    error("gmm_jump_problem not implemented with constraints for $(typeof(GMMModel))")
  end
  f = obj(model)
  JuMP.register(m, :obj, d, (θ...)->f([θ...]), autodiff=true)
  @NLobjective(m, Min, obj(θ...))
  return(m)
end

 
"""
    gel_optim_args(model::GMMModel, h::Function=log)

Return tuple, `out` for calling `optimize(out..., IPNewton())` afterward. 

It seems that IPNewton() works better if the constraint on p is 0 ≤
sum(p) ≤ 1 instead of sum(p)=1, and you begin the optimizer from a
point with sum(p) < 1. 
"""
function gel_optim_args(model::GMMModel, gel::Function=log)
  gi = get_gi(model)
  n = number_observations(model)
  d = number_parameters(model)
  k = number_moments(model)
  el(x) = -sum(gel.(x[1:n]))
  ∇el!(g,x) = (g.= 0; g[1:n] .= -(p->ForwardDiff.derivative(gel,p)).(x[1:n]); g)
  ∇²el! = function (h,x)
    h .= 0
    for i in 1:n
      h[i,i] = -ForwardDiff.derivative(p->ForwardDiff.derivative(gel,p),x[i])
    end
    h
  end
  cfn = gmm_constraints(model)
  nc = cfn===nothing ? 0 : length(cfn(zeros(d)))
  con_c! = function(c,x)
    p = x[1:n]
    θ = x[(n+1):end]
    c[1] = sum(p)
    c[2:(k+1)] .= sum(gi(θ).*p, dims=1)[:]
    if cfn!==nothing      
      c[(k+2):(k+nc+1)] .= cfn(θ)
    end
    c
  end
  con_jac! = function(J,x)
    p = x[1:n]
    θ = x[(n+1):end]
    J[1,1:n] .= 1
    J[1,(n+1):end] .= 0
    J[2:(k+1),1:n] .= gi(θ)'
    J[2:(k+1),(n+1):end] .= ForwardDiff.jacobian(t->sum(gi(t).*p, dims=1), θ)
    if (cfn!==nothing)
      J[(k+2):(k+nc+1),1:n].=0
      J[(k+2):(k+nc+1),(n+1):end].=ForwardDiff.jacobian(cfn, θ)
    end
    J
  end
  con_hess! = function(h,x,λ)
    p = x[1:n]
    θ = x[(n+1):end]
    λDgi = ForwardDiff.jacobian(x->gi(x)*λ[2:(1+k)], θ)
    for i in 1:n
      for m in 1:d
        h[i,(n+m)] += λDgi[i,m]
        h[(n+m),i] += λDgi[i,m]
      end
    end
    h[(n+1):(n+d),(n+1):(n+d)] .+=
      ForwardDiff.hessian(t->dot(λ[2:(1+k)],sum(gi(t).*p, dims=1)), θ) 
    if (cfn!==nothing)
      h[(n+1):(n+d),(n+1):(n+d)] .+= ForwardDiff.hessian(t->dot(λ[(2+k):end],cfn(t)), θ)
    end
    h
  end
  
  lcon = [0.0, zeros(k+nc)...]
  ucon = [1.0, zeros(k+nc)...]
  lvar = [zeros(n)... , fill(-Inf,d)...]
  uvar = [ones(n)... ,  fill(Inf,d)...]
  x0 = [fill(1/(1.1*n),n)... , zeros(d)...]
  objective = TwiceDifferentiable(el, ∇el!, ∇²el!, x0)  
  constraints = TwiceDifferentiableConstraints(con_c!, con_jac!, con_hess!,
                                               lvar, uvar, lcon, ucon)    
  return((objective=objective,constraints=constraints,x0=x0))
end
