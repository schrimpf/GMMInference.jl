# Extremum Estimation

```math
\def\indep{\perp\!\!\!\perp}
\def\Er{\mathrm{E}}
\def\R{\mathbb{R}}
\def\En{{\mathbb{E}_n}}
\def\Pr{\mathrm{P}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\def\inprob{\,{\buildrel p \over \rightarrow}\,}
\def\indist{\,{\buildrel d \over \rightarrow}\,}
\,
```

Many, perhaps most, estimators in econometrics are extrumem
estimators. That is, many estimators are defined by

```math
\hat{\theta} = \argmax_{\theta \in \Theta}
\hat{Q}_n(\theta)
```

where ``\hat{Q}_n(\theta)`` is some objective
function that depends on data. Examples include maximum likelihood,

```math
\hat{Q}_n(\theta) = \frac{1}{n} \sum_{i=1}^n f(z_i | \theta)
```

GMM,

```math
\hat{Q}_n(\theta) = \left(\frac{1}{n} \sum_{i=1}^n g(z_i,
\theta)\right)' \hat{W} \left(\frac{1}{n} \sum_{i=1}^n g(z_i,
\theta)\right)
```

and nonlinear least squares

```math
\hat{Q}_n(\theta) =
\frac{1}{n} \sum_{i=1}^n (y_i - h(x_i,\theta))^2.
```

See [newey1994](@cite) for more details and examples.

## Example: logit

As a simple example, let's look look at some code for estimating a
logit.

```@example ee
using Distributions, Optim, BenchmarkTools
import ForwardDiff
function simulate_logit(observations, β)
  x = randn(observations, length(β))
  y = (x*β + rand(Logistic(), observations)) .>= 0.0
  return((y=y,x=x))
end

function logit_likelihood(β,y,x)
  p = map(xb -> cdf(Logistic(),xb), x*β)
  sum(log.(ifelse.(y, p, 1.0 .- p)))
end

n = 500
k = 3
β0 = ones(k)
(y,x) = simulate_logit(n,β0)
Q = β -> -logit_likelihood(β,y,x)
Q(β0)
```

Now we maximize the likelihood using a few different algorithms from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)

```@example ee
@btime nm=optimize(Q, zeros(k), NelderMead())
@btime bfgs=optimize(Q, zeros(k), BFGS(), autodiff = :forward)
@btime ntr=optimize(Q, zeros(k), NewtonTrustRegion(), autodiff =:forward);
```

### Aside: Reverse mode automatic differentiation

For functions ``f:\R^n \to \R^m``, the work for forward automatic
differentiation increases linearly with ``n``. This is because forward
automatic differentiation applies the chain rule to each of the ``n``
inputs. An alternative, is reverse automatic differentiation. Reverse
automatic differentiation is also based on the chain rule, but it
works backward from ``f`` through intermediate steps back to ``x``. The
work needed here scales linearly with ``m``. Since optimization problems
have ``m=1``, reverse automatic differentiation can often work well. The
downsides of reverse automatic differentiation are that: (1) it can
require a large amount of memory and (2) it is more difficult to
implement. There are handful of Julia packages that provide reverse
automatic differentiation, but they have some limitations in terms of
what functions thay can differentiate. Flux.jl and Zygote.jl are two such packages.


```@example ee
using Optim, BenchmarkTools
import Zygote
dQr = β->Zygote.gradient(Q,β)[1]
dQf = β->ForwardDiff.gradient(Q,β)

@show dQr(β0) ≈ dQf(β0)

@btime dQf(β0)
@btime dQr(β0)

n = 500
k = 200
β0 = ones(k)
(y,x) = simulate_logit(n,β0)
Q = β -> -logit_likelihood(β,y,x)
dQr = β->Zygote.gradient(Q,β)[1]
dQf = β->ForwardDiff.gradient(Q,β)
@show dQr(β0) ≈dQf(β0)
@btime dQf(β0);
@btime dQr(β0);
```

# Review of extremum estimator theory

This is based on [newey1994](@cite). You should already be familiar with this
from 627, so we will just state some basic "high-level" conditions for
consistency and asymptotic normality.

## Consistency


!!! tip "Theorem: consistency for extremum estimators"

    Assume

    1. ``\hat{Q}_n(\theta)`` converges uniformly in probability to
    ``Q_0(\theta)``

    2. ``Q_0(\theta)`` is uniquely maximized at ``\theta_0``.

    3. ``\Theta`` is compact and ``Q_0(\theta)`` is continuous.

    Then ``\hat{\theta} \to^p \theta_0``


## Asymptotic normality


!!! tip "Theorem: asymptotic normality for extremum estimators"

    Assume

    1. ``\hat{\theta} \to^p \theta_0``

    2. ``\theta_0 \in interior(\Theta)``

    3. ``\hat{Q}_n(\theta)`` is twice continuously differentiable in
    open ``N`` containing ``\theta`` , and
    ``\sup_{\theta \in N} \left\Vert \nabla^2 \hat{Q}_n(\theta) - H(\theta) \right\Vert \to^p 0``
    with ``H(\theta_0)`` nonsingular

    4. ``\sqrt{n} \nabla \hat{Q}_n(\theta_0) \leadsto N(0,\Sigma)``

    Then ``\sqrt{n} (\hat{\theta} - \theta_0) \leadsto N\left(0,H^{-1} \Sigma H^{-1} \right)``

Implementing this in Julia using automatic differentiation is straightforward.

```@example ee
function logit_likei(β,y,x)
  p = map(xb -> cdf(Logistic(),xb), x*β)
  log.(ifelse.(y, p, 1.0 .- p))
end

function logit_likelihood(β,y,x)
  mean(logit_likei(β,y,x))
end

n = 100
k = 3
β0 = ones(k)
(y,x) = simulate_logit(n,β0)

Q = β -> -logit_likelihood(β,y,x)
optres = optimize(Q, zeros(k), NewtonTrustRegion(), autodiff =:forward)
βhat = optres.minimizer

function asymptotic_variance(Q,dQi, θ)
  gi = dQi(θ)
  Σ = gi'*gi/size(gi)[1]
  H = ForwardDiff.hessian(Q,θ)
  invH = inv(H)
  (variance=invH*Σ*invH, Σ=Σ, invH=invH)
end

avar=asymptotic_variance(θ->logit_likelihood(θ,y,x),
                         θ->ForwardDiff.jacobian(β->logit_likei(β,y,x),θ),βhat)
@show avar.variance/n
@show -avar.invH/n
@show inv(avar.Σ)/n
```

For maximum likelihood, the information equality says ``-H = \Sigma``,
so the three expressions above have the same probability limit, and
are each consistent estimates of the variance of ``\hat{\theta}``.

The code above is for demonstration and learning. If we really wanted
to estimate a logit for research, it would be better to use a
well-tested package. Here's how to estimate  a logit using GLM.jl.

```@example ee
using GLM, DataFrames
df = DataFrame(x, :auto)
df[!,:y] = y
glmest=glm(@formula(y ~ -1 + x1+x2+x3), df, Binomial(),LogitLink())
@show glmest
@show vcov(glmest)
```

## Delta method
    
In many models, we are interested in some transformation of the
parameters in addition to the parameters themselves. For example, in a
logit, we might want to report marginal effects in addition to the
coefficients. In structural models, we typically use the parameter
estimates to conduct counterfactual simulations. In many
situations we are more interested these transformation(s) of
parameters than in the parameters themselves. The delta method is one
convenient way to approximate the distribution of transformations of
the model parameters.

!!! tip "Theorem: Delta method"

    Assume:

    1. ``\sqrt{n} (\hat{\theta} - \theta_0) \leadsto N(0,\Omega)``

    2. ``g: \R^k \to \R^m`` is continuously differentiable

    Then ``\sqrt{n}(g(\hat{\theta}) - g(\theta_0)) \leadsto N(0, \nabla g(\theta_0)^T \Omega \nabla g(\theta_0)``

The following code uses the delta method to plot a 90% pointwise
confidence band around the estimate marginal effect of one of the
regressors.

```@example ee
using LinearAlgebra
function logit_mfx(β,x)
  out = ForwardDiff.jacobian(x-> map(xb -> cdf(Logistic(),xb), x*β), x)
  out = reshape(out, size(out,1), size(x)...)
end

function delta_method(g, θ, Ω)
  dG = ForwardDiff.jacobian(θ->g(θ),θ)
  dG*Ω*dG'
end

nfx = 100
xmfx = zeros(nfx,3)
xmfx[:,1] .= -3.0:(6.0/(nfx-1)):3.0

mfx = logit_mfx(βhat,xmfx)
vmfx = delta_method(β->diag(logit_mfx(β,xmfx)[:,:,1]), βhat, avar.variance/n)
sdfx = sqrt.(diag(vmfx))

using Plots, LaTeXStrings
plot(xmfx[:,1],diag(mfx[:,:,1]),ribbon=quantile(Normal(),0.95)*sdfx,fillalpha=0.5,
     xlabel=L"x_1", ylabel=L"\frac{\partial}{\partial x_1}P(y=1|x)", 
     legend=false,title="Marginal effect of x[1] when x[2:k]=0")
```

The same approach can be used to compute standard errors and
confidence regions for the results of more complicated counterfactual
simulations, as long as the associated simulations are smooth
functions of the parameters. However, sometimes it might be more
natural to write simulations with outcomes that are not smooth in the
parameters. For example, the following code uses simulation to
calculate the change in the probability of ``y`` from adding 0.1 to
``x``.

```@example ee
function counterfactual_sim(β, x, S)
  function onesim()
    e = rand(Logistic(), size(x)[1])
    baseline= (x*β .+ e .> 0)
    counterfactual= ((x.+0.1)*β .+ e .> 0)
    mean(counterfactual.-baseline)
  end
  mean([onesim() for s in 1:S])
end
@show ∇s = ForwardDiff.gradient(β->counterfactual_sim(β,x,10),βhat)
```

Here, the gradient is 0 because the simulation function is a
step-function. In this situation, an alternative to the delta method
is the simulation based approach of [krinsky1986](@cite). The procedure is
quite simple. Suppose
``\sqrt{n}(\hat{\theta} - \theta_0) \leadsto N(0,\Omega)``,
and you want to an estimate of the distribution of ``g(\theta)``.
Repeatedly draw ``\theta_s \sim N(\hat{\theta}, \Omega/n)`` and compute
``g(\theta_s)``. Use the distribution of ``g(\theta_s)`` for
inference. For example, a 90% confidence interval for ``g(\theta)``
would be the 5%-tile of ``g(\theta_s)`` to the 95%-tile of
``g(\theta_s)``.

```@example ee
Ω = avar.variance/n
Ω = Symmetric((Ω+Ω')/2) # otherwise, it's not exactly symmetric due to
                        # floating point roundoff
function kr_confint(g, θ, Ω, simulations; coverage=0.9)
  θs = [g(rand(MultivariateNormal(θ,Ω))) for s in 1:simulations]
  quantile(θs, [(1.0-coverage)/2, coverage + (1.0-coverage)/2])
end

@show kr_confint(β->counterfactual_sim(β,x,10), βhat, Ω, 1000)

# a delta method based confidence interval for the same thing
function counterfactual_calc(β, x)
  baseline      = cdf.(Logistic(), x*β)
  counterfactual= cdf.(Logistic(), (x.+0.1)*β)
  return([mean(counterfactual.-baseline)])
end
v = delta_method(β->counterfactual_calc(β,x), βhat, Ω)
ghat = counterfactual_calc(βhat,x)
@show [ghat + sqrt(v)*quantile(Normal(),0.05), ghat +
       sqrt(v)*quantile(Normal(),0.95)]
```

