######################################################################
# TODO: Take methods from docs/jmd/extremumEstimation.jmd and put here
######################################################################

######################################################################

# EL inference 

"""
    gel_pλ(model::GMMModel, h::Function=log)

Return a function that given parameters θ solves

`` max_p sum h(p) s.t. sum(p) = 1, sum(p*gi(θ)) = 0 ``

The returned function gives (p(θ),λ(θ))

The returned function is not thread-safe.  
"""
function gel_pλ(model::GMMModel, h::Function=log)  
  n = number_observations(model)
  d = number_parameters(model)
  k = number_moments(model)
  gi = get_gi(model)
  g = gi(zeros(d))
  p = Convex.Variable(n)
  el = sum(h.(p))
  problem = Convex.maximize(el, sum(p)==1, p>=0, g'*p==0) 
  function(θ)
    g = gi(θ)
    problem.constraints[3] =  g'*p==0
    optimizer = optimizer_with_attributes(ECOS.Optimizer, "verbose"=>false)
    Convex.solve!(problem, optimizer, #warmstart=false,
                  verbose=false)
    (p=p.value ,λ=problem.constraints[3].dual)
  end
end

""" 
    gel_tests(θ, model::GMMModel, h::Function=log)

Computes GEL test statistics for H₀ : θ = θ₀. Returns a tuple
containing statistics and p-values.
"""
function gel_tests(θ, model::GMMModel, h::Function=log)
  d = number_parameters(model)
  k = number_moments(model)
  p, λ = gel_pλ(model, h)(θ)
  # Compared to Andrews & Guggenberger, this λ = -n*λ(AG)
  n = length(p)
  λ /= -n
  lr = -2*sum(h.(p) .- h(1/n))
  
  gi = get_gi(model)
  iΔ = inv(cov(gi(θ)))
  D = ForwardDiff.jacobian(x->gi(x)'*p, θ)
  S = (n*λ'*D*inv(D'*iΔ*D)*D'*λ)[1]
  return(GELR=lr, S=S, plr=cdf(Chisq(k),lr), ps=cdf(Chisq(d),S))
end


