# Empirical likelihood

An interesting alternative to GMM is (generalized) empirical
likelihood (GEL). Empirical likelihood has some appealing higher-order
statistical properties. In particular, it can be shown to have lower
higher order asymptotic bias than GMM. See [newey2004](@cite). Relatedly, certain test statistics based on EL are
robust to weak identification [guggenberger2005](@cite). In fact, the identification robust tests
that we have discusses are all based on the CUE-GMM objective
function. The CUE-GMM objetive is a special case of generalized
empirical likelihood.

A perceived downside of GEL is that it involves a more difficult
looking optimization problem than GMM. However, given the ease with
which Julia can solve high dimensional optimization problems, GEL is
very feasible. 

As in the extremum estimation notes, suppose we have moment conditions
such that
```math
\mathrm{E}[g_i(\theta)] = 0
```
where ``g_i:\R^d \to \R^k`` are some data dependent moment
conditions. The empirical likelihood estimator solves
```math
\begin{align*}
    (\hat{\theta}, \hat{p}) = & \argmax_{\theta,p} \frac{1}{n} \sum_i
    \log(p_i) \;\; s.t.  \\
     & \sum_i p_i = 1, \;\; 0\leq p_i \leq 1 \\
     & \sum_i p_i g_i(\theta) = 0 
\end{align*}
```

Generalized empirical likelihood replaces ``\log(p)`` with some other
convex function ``h(p)``, 
```math
\begin{align*}
    (\hat{\theta}^{GEL,h}, \hat{p}) = & \argmin_{\theta,p}
    \frac{1}{n}\sum_i h(p_i) \;\; s.t.  \\
     & \sum_i p_i = 1, \;\; 0\leq p \leq 1 \\
     & \sum_i p_i g_i(\theta) = 0 
\end{align*}
```
setting ``h(p) = \frac{1}{2}(p^2-(1/n)^2)`` results in an estimator
identical to the CUE-GMM estimator.

A common approach to computing GEL estimators is to eliminate ``\pi`` by
looking at the dual problem
```math
\hat{\theta}^{GEL}  = \argmin_{\theta}\sup_\lambda \sum_i \rho(\lambda'g_i(\theta))
```
where ``\rho`` is some function related to ``h``. See [newey2004](@cite) for
details. There can be some analytic advantages to doing so, but
computationally, the original statement of the problem has some
advantages. First, there is more existing software for solving
constrained minimization problems than for solving saddle point
problems. Second, although ``p`` is high dimensional, it enters the
constraints linearly, and the objective function is concave. Many
optimization algorithms will take good advantage of this. 

Let's look at some Julia code. Since the problem involves many
variables with linear constraints, it is worthwhile to use JuMP for
optimization. The code is slightly more verbose, but the speed of
JuMP (and the Ipopt solver) are often worth it.

```@example el
using GMMInference, JuMP, Ipopt, LinearAlgebra, Distributions 
import Random

n = 300
d = 4
k = 2*d
β0 = ones(d)
π0 = vcat(I,ones(k-d,d))
ρ = 0.5
Random.seed!(622)
data = IVLogitShare(n, β0, π0, ρ);
```

```@example el
# set up JuMP problem
Ty = quantile.(Logistic(),data.y)   
m = Model()
@variable(m, 0.0 <= p[1:n] <= 1.0)
@variable(m, θ[1:d])
@constraint(m, prob,sum(p)==1.0)
@constraint(m, momentcon[i=1:k], dot((Ty - data.x*θ).*data.z[:,i],p)==0.0)
@NLobjective(m,Max, sum(log(p[i]) for i in 1:n))
m
```
The `gel_jump_problem` function from `GMMInference.jl` does the same
thing as the above code cell. 

Let's solve the optimization problem.
```@example el
set_optimizer(m, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5))
set_start_value.(m[:θ], 0.0)
set_start_value.(m[:p], 1/n)
optimize!(m)
@show value.(m[:θ])
@show value.(m[:p][1:10])
```

For comparison here is how long it takes JuMP + Ipopt to solve for the
CUE-GMM estimator. 

```@example el
@show mcue = gmm_jump_problem(data, cue_objective)
set_start_value.(mcue[:θ], 0.0)
set_optimizer(mcue,  optimizer_with_attributes(Ipopt.Optimizer, "print_level" =>5))
optimize!(mcue)
@show value.(mcue[:θ]) 
```

In this comparison, EL is both faster and more robust to initial
values than CUE-GMM. GMM with a fixed weighting matrix will likely be
faster than either.

## GEL with other optimization packages

We can also estimate GEL models with other optimization
packages. Relative to JuMP, other optimization packages have the
advantage that the problem does not have to be written in any special
syntax. However, other packages have the downside that they will not
recognize any special structure in the constraints (linear, quadratic,
sparse, etc) unless we explicitly provide it. Let's see how much, if
any difference this makes to performance. 

Here we will use the `NLPModels.jl` interface to
Ipopt. Essentially, all this does is call `ForwardDiff` on the
objective function and constraints, and then give the resulting
gradient and hessian functions to Ipopt.

```@example el
using NLPModelsIpopt
gel = gel_nlp_problem(data)
ip = ipopt(gel)
```

As you see, this approach is far slower than with JuMP. Notice that
the number of iterations and function evaluations are identical. The
big difference is that JuMP evaluates the function (and its
derivatives) very quickly, while NLP takes much much longer. I would
guess that this is largely because it is using ForwardDiff to
calculate a gradients and hessians for 304 variables.

Let's also estimate the model using `Optim.jl`. 

```@example el
using Optim
args = gel_optim_args(data)
@time opt=optimize(args[1],args[2],
                   [fill(1/(1.1*n),n)..., zeros(d)...],
                   IPNewton(μ0=:auto, show_linesearch=false),
                   Optim.Options(show_trace=true,
                                 allow_f_increases=true,
                                 successive_f_tol=10,
                                 allow_outer_f_increases=true))
@show opt.minimizer[(n+1):end]
```

The `IPNewton()` optimizer from `Optim.jl` appears to be much less
efficient than `Ipopt`. `IPNewton` takes more than 4 times as many
iterations. 

# Inference for EL

[guggenberger2005](@cite) show that GEL
versions of the AR and LM statistics are robust to weak
identification. The GEL version of the AR statistic is the generalized
empirical likelihood ratio. Specifically, [guggenberger2005](@cite) show that

```math
GELR(\theta_0) = 2\sum_{i=1}^n\left(h(p_i(\theta_0))  -
   h(1/n)\right) \leadsto \chi^2_k 
```

where ``p_i(\theta_0)`` are given by

```math
\begin{align*}
    p(\theta)  =  \argmax_{0 \leq p \leq 1} \sum h(p_i) \text{ s.t. } & \sum_i p_i
    = 1 \\
& \sum_i p_i g_i(\theta) = 0
\end{align*}
```

The GELR statistic shares the downsides of the AR statistic --- the
degrees of freedom is the number of moments instead of the number of
parameters, which tends to lead to lower power in overidentified
models; and it combines a test of misspecification with a location
test for ``\theta``. 

Consequently, it can be useful to instead look at a Lagrange
multiplier style statistic. The true ``\theta`` maximizes the
empirical likelihood, so 

```math
0 = \sum_{i=1}^n \nabla_\theta h(p_i(\theta_0)) = \lambda(\theta_0)' \sum_{i=1}^n
p_i(\theta_0) \nabla_\theta g_i(\theta_0) \equiv \lambda(\theta_0) D(\theta_0)
```

where ``p_i(\theta_0)`` is as defined above, and ``\lambda(\theta_0)`` are
the mulitpliers on the empirical moment condition constraint. Andrews
and Guggenberger show that a quadratic form in the above score
equation is asymptotically ``\chi^2_d``. To be specific, let
``\Delta(\theta) = E[(1/n\sum_i g_i(\theta) - E[g(\theta)])(1/n \sum_i
 g_i(\theta) - E[g(\theta)])']``  and define

```math
S(\theta) = n\lambda(\theta)' D(\theta) \left( D(\theta)
\Delta(\theta)^{-1} D(\theta) \right)^{-1} D(\theta)'\lambda(\theta)
```

then ``S(\theta_0) \leadsto \chi^2_d``. This result holds whether or not
``\theta`` is strongly identified. 

### Implementation

Computing the ``GELR`` and ``S`` statistics requires solving a linear
program for each ``\theta`` we want to test. Fortunately, linear
programs can be solved very quickly. See `gel_pλ` in `GMMInference.jl`
for the relevant code. 

Let's do a simulation to check that these tests have correct coverage.

```@example el
using Plots
Plots.gr()
S = 500
n = 200
d = 2
β0 = ones(d)
ρ = 0.5
k = 3
function sim_p(π0)
  data = IVLogitShare(n, β0, π0, ρ)
  GELR, S, plr, ps = gel_tests(β0, data)
  [plr ps]
end
πweak = ones(k,d) .+ vcat(diagm(0=>fill(0.001,d)),zeros(k-d,d))  
πstrong = vcat(5*diagm(0=>ones(d)),ones(k-d,d)) 
pweak=vcat([sim_p(πweak ) for s in 1:S]...)
pstrong=vcat([sim_p(πstrong) for s in 1:S]...)

pgrid = 0:0.01:1
plot(pgrid, p->(mean( pstrong[:,1] .<= p)-p), legend=:bottomleft,
     label="GELR, strong ID", style=:dash, color=:blue,
     xlabel="p", ylabel="P(p value < p)-p",
     title="Simulated CDF of p-values - p")  
plot!(pgrid, p->(mean( pstrong[:,2] .<= p)-p),
      label="S, strong ID", style=:dash, color=:green)

plot!(pgrid, p->(mean( pweak[:,1] .<= p)-p),
      label="GELR, weak ID", style=:solid, color=:blue)
plot!(pgrid, p->(mean( pweak[:,2] .<= p)-p),
      label="S, weak ID", style=:solid, color=:green)
plot!(pgrid,0,alpha=0.5, label="")
```

### Subvector inference

[guggenberger2005](@cite) also give results for
subvector inference. Let ``(\alpha, \beta) =\theta``. Assume ``\beta`` is
strongly identified. Guggenberger and Smith show that analogs of
``GELR`` and ``S`` with ``\beta`` concentrated out lead to valid tests for
``\alpha``, whether ``\alpha`` is weakly or strongly identified. 


## Bootstrap for EL

For bootstrapping GMM, we discussed how it is important that the null
hypothesis holds in the bootstrapped data. In GMM we did this by
substracting the sample averages of the moments. In GEL, an
alternative way to impose the null, is to sample the data with
probabilities ``\hat{p}_i`` instead of with equal proability. See
[brown2002](@cite) for more information. 
