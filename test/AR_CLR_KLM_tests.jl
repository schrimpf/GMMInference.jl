"""
This test is the using the same procedure as identificationRobustInference.jmd 
with using new funciton which have been added to scr/HypothesisTests. 
For compating, the same methods in identificationRobustInference.jmd also
has been presented.
"""


using Optim, ForwardDiff, LinearAlgebra, Distributions, LaTeXStrings
"""
    AR_gmm_obj(gi:Function)

Returns a function `ar(θ)` where:
``ar(θ) = n [1/n \\sum g_i(θ)]' \\widehat{Var}(g_i(θ))^{-1}[1/n \\sum g_i(θ)] ``
"""
function AR_gmm_obj(gi::Function)
  function(θ)
    g = gi(θ)
    m = mean(g,dims=1)
    W = inv(cov(g))
    (size(g)[1]*( m*W*m')[1])
  end
end
"""
    AR_test(n::Integer, β::AbstractVector,
                      π::AbstractMatrix, ρ)

Returns a function `ar(θ)` where:
``ar(θ) = n [1/n \\sum g_i(θ)]' \\widehat{Var}(g_i(θ))^{-1}[1/n \\sum g_i(θ)] ``

Or equivalent to:
``AR_gmm_obj(get_gi(IVLogitShare(n,β0,π0,ρ)))``
"""
function AR_test(n::Integer, β::AbstractVector,
                      π::AbstractMatrix, ρ)
    ivlogitshare = IVLogitShare(n,β,π,ρ)
    gi = get_gi(ivlogitshare)
    return AR_gmm_obj(gi)
end


"""
There are also identification robust versions of likelihood ratio and
lagrange multiplier test. Moreire (2003)[@moreira2003] proposed a conditional
likelihood ratio (CLR) test for weakly identified linear IV
models. Kleibergen (2005)[@kleibergen2005] developed a Lagrange multiplier (often called
the KLM) test and extended Moreira's CLR test to weakly identified GMM
models.  More recently, Andrews and Guggenberge (2015) [@andrews2015]
and Andrews and Guggenberge (2017) [@andrews2017] showed the
validity of these tests under more general conditions. These tests are
somewhat more complicated than the AR test, but they have the
advantage that they are often more powerful. The AR test statistic has
a $\chi^2_{m}$ distribution, where $m$ is the number of moments. The
CLR and KLM statistics under strong identification have $\chi^2_k$
distributions (as does the Wald statistic), where $k$ is the number of
parameters. Consequently, when the model is overidentified, the CLR
and LM tests are more powerful than the AR test. 


Here is an implementation of the KLM and CLR statistics. The names of
variables roughly follows the notation of Andrews and
Guggenberge(2017

"""

"""
    statparts(gi::Function)

Return a function.
    
compute common components of klm, rk, & clr stats
follows notation of Andrews & Guggenberger 2017, section 3.1
"""
function statparts(gi::Function)
  # compute common components of klm, rk, & clr stats
  # follows notation of Andrews & Guggenberger 2017, section 3.1
    function P(A::AbstractMatrix) # projection matrix
    A*pinv(A'*A)*A'
    end
    function(θ)
        giθ = gi(θ)
        p = length(θ)    
        (n, k) = size(giθ)
        Ω = cov(giθ)  
        gn=mean(gi(θ), dims=1)'
        #G = ForwardDiff.jacobian(θ->mean(gi(θ),dims=1),θ)
        Gi= ForwardDiff.jacobian(gi,θ)
        Gi = reshape(Gi, n , k, p)
        G = mean(Gi, dims=1)
        Γ = zeros(eltype(G),p,k,k)
        D = zeros(eltype(G),k, p)
        for j in 1:p
        for i in 1:n
          Γ[j,:,:] += (Gi[i,:,j] .- G[1,:,j]) * giθ[i,:]'
        end
        Γ[j,:,:] ./= n
        D[:,j] = G[1,:,j] - Γ[j,:,:]*inv(Ω)*gn
        end
        return(n,k,p,gn, Ω, D, P)
    end
end

"""
    klm(gi::Function)

Return test statistic for KLM method.
"""

function klm(gi::Function)
    SP = statparts(gi)
    function(θ)
        (n,k,p,gn, Ω, D, P) = SP(θ)
        return n*(gn'*Ω^(-1/2)*P(Ω^(-1/2)*D)*Ω^(-1/2)*gn)[1]
    end
end


"""
    clr(gi::Function)

Return test statistic for CLR method.
"""
function clr(gi::Function)
  SP = statparts(gi)
  function(θ)
        (n,k,p,gn, Ω, D, P) = SP(θ)
        rk = eigmin(n*D'*inv(Ω)*D)
        AR  = (n*gn'*inv(Ω)*gn)[1]
        lm = (n*(gn'*Ω^(-1/2)*P(Ω^(-1/2)*D)*Ω^(-1/2)*gn))[1]  
        lr = 1/2*(AR - rk + sqrt( (AR-rk)^2 + 4*lm*rk))

        # simulate to find p-value
        S = 5000
        function randc(k,p,r,S)
        χp = rand(Chisq(p),S)
        χkp = rand(Chisq(k-p),S)
        0.5.*(χp .+ χkp .- r .+
              sqrt.((χp .+ χkp .- r).^2 .+ 4 .* χp.*r))
        end
        csim = randc(k,p,rk,S)
        return mean(csim.<=lr)
    end
end



function plot_cr(β,V, tests::AbstractArray{Function}, labels; ngrid=30)
  lb = β - sqrt.(diag(V))*5
  ub = β + sqrt.(diag(V))*5
  fig=scatter([β0[1]], [β0[2]], markersize=8, legend=false,
              xlabel=L"\beta_1", ylabel=L"\beta_2")
  ntest = 1000
  βtest = [rand(2).*(ub-lb) .+ lb for i in 1:ntest]
  pval = tests[1].(βtest)
  βtest = vcat(βtest'...)
  crit = 0.9
  fig=scatter!(βtest[:,1],βtest[:,2], group=(pval.<crit), legend=false,
               markersize=4, markerstrokewidth=0.0, seriesalpha=0.5,
               palette=:heat)
  b1 = lb[1]:(ub[1]-lb[1])/ngrid:ub[1]
  b2 = lb[2]:(ub[2]-lb[2])/ngrid:ub[2]
  colors = [:black, :red, :blue, :green]
  for t in 1:length(tests)
    fig=contour!(b1,b2,(a,b)->tests[t]([a,b]),
             levels = [0.9, 0.95],
             contour_labels=false, legend=false,
             label = labels[t],
             c=cgrad([colors[t],colors[t]],[0.0,1.0]))
  end
  fig
end



function gmmVar(θ,gi,W)
  g = gi(θ)
  n = size(g)[1]
  D = ForwardDiff.jacobian(θ->mean(gi(θ),dims=1),θ)
  Σ = cov(gi(θ))
  1/n*inv(D'*W*D)*(D'*W*Σ*W*D)*inv(D'*W*D)
end
function gmmObj(θ,gi,W)
  g = gi(θ)
  m = mean(g,dims=1)
  (size(g)[1]*( m*W*m')[1]) # return scalar, not 1x1 array
end



n = 50
k = 2
iv =3 
π0 = vcat(0.1*diagm(0=>ones(k)),0.2*ones(iv-k,k)) 
ρ = 0.5  
data = IVLogitShare(n,β0,π0,ρ)
opt1 = optimize(θ->gmmObj(θ, get_gi(data) ,I),
                β0, BFGS(), autodiff =:forward)
β1 = opt1.minimizer
V1 = gmmVar(β1,get_gi(data),I)
  
KLM_f = klm(get_gi(data))
pklm = θ -> cdf(Chisq(length(β1)), KLM_f(θ) )

AR_f = AR_gmm_obj(get_gi(data))
par  = θ ->cdf(Chisq(size(data.z)[2]), AR_f(θ) ) 
pclr  = clr(get_gi(data))
pwald = θ -> cdf(Chisq(length(β1)),(θ-β1)'*inv(V1)*(θ-β1))
plot_cr(β1,V1, [pclr, pklm, par, pwald],
        ["CLR","KLM","AR","Wald"], ngrid=40)




"""
Now the equivalent codes from part of identificationRobustInference.jmd 
For comparison
"""
function simulate_ivshare(n,β,γ,ρ)
  z = randn(n, size(γ)[1])
  endo = randn(n, length(β))
  x = z*γ .+ endo
  ξ = rand(Normal(0,sqrt((1.0-ρ^2))),n).+endo[:,1]*ρ 
  y = cdf.(Logistic(), x*β .+ ξ)
  return((y=y,x=x,z=z))  
end
n = 100
k = 2
iv = 3
β0 = ones(k)
π0 = vcat(5*I,ones(iv-k,k)) 
ρ = 0.5  
(y,x,z) = simulate_ivshare(n,β0,π0,ρ)

function gi_ivshare(β,y,x,z)
  ξ = quantile.(Logistic(),y) .- x*β
  ξ.*z
end


function gmmObj(θ,gi,W)
  g = gi(θ)
  m = mean(g,dims=1)
  (size(g)[1]*( m*W*m')[1]) # return scalar, not 1x1 array
end

function gmmVar(θ,gi,W)
  g = gi(θ)
  n = size(g)[1]
  D = ForwardDiff.jacobian(θ->mean(gi(θ),dims=1),θ)
  Σ = cov(gi(θ))
  1/n*inv(D'*W*D)*(D'*W*Σ*W*D)*inv(D'*W*D)
end

function ar(θ,gi)
  gmmObj(θ,gi,inv(cov(gi(θ))))
end



using ForwardDiff, Plots, Optim
Plots.gr()
function statparts(θ,gi)
  # compute common components of klm, rk, & clr stats
  # follows notation of Andrews & Guggenberger 2017, section 3.1
  function P(A::AbstractMatrix) # projection matrix
    A*pinv(A'*A)*A'
  end
  giθ = gi(θ)
  p = length(θ)    
  (n, k) = size(giθ)
  Ω = cov(giθ)  
  gn=mean(gi(θ), dims=1)'
  #G = ForwardDiff.jacobian(θ->mean(gi(θ),dims=1),θ)
  Gi= ForwardDiff.jacobian(gi,θ)
  Gi = reshape(Gi, n , k, p)
  G = mean(Gi, dims=1)
  Γ = zeros(eltype(G),p,k,k)
  D = zeros(eltype(G),k, p)
  for j in 1:p
    for i in 1:n
      Γ[j,:,:] += (Gi[i,:,j] .- G[1,:,j]) * giθ[i,:]'
    end
    Γ[j,:,:] ./= n
    D[:,j] = G[1,:,j] - Γ[j,:,:]*inv(Ω)*gn
  end
  return(n,k,p,gn, Ω, D, P)
end

function klm(θ,gi)
  (n,k,p,gn, Ω, D, P) = statparts(θ,gi)
  lm = n*(gn'*Ω^(-1/2)*P(Ω^(-1/2)*D)*Ω^(-1/2)*gn)[1]
end

function clr(θ,gi)
  (n,k,p,gn, Ω, D, P) = statparts(θ,gi)
  
  rk = eigmin(n*D'*inv(Ω)*D)
  AR  = (n*gn'*inv(Ω)*gn)[1]
  lm = (n*(gn'*Ω^(-1/2)*P(Ω^(-1/2)*D)*Ω^(-1/2)*gn))[1]  
  lr = 1/2*(AR - rk + sqrt( (AR-rk)^2 + 4*lm*rk))
  
  # simulate to find p-value
  S = 5000
  function randc(k,p,r,S)
    χp = rand(Chisq(p),S)
    χkp = rand(Chisq(k-p),S)
    0.5.*(χp .+ χkp .- r .+
          sqrt.((χp .+ χkp .- r).^2 .+ 4 .* χp.*r))
  end
  csim = randc(k,p,rk,S)
  pval = mean(csim.<=lr)
end


function plot_cr(β,V, tests::AbstractArray{Function}, labels; ngrid=30)
  lb = β - sqrt.(diag(V))*5
  ub = β + sqrt.(diag(V))*5
  fig=scatter([β0[1]], [β0[2]], markersize=8, legend=false,
              xlabel=L"\beta_1", ylabel=L"\beta_2")
  ntest = 1000
  βtest = [rand(2).*(ub-lb) .+ lb for i in 1:ntest]
  pval = tests[1].(βtest)
  βtest = vcat(βtest'...)
  crit = 0.9
  fig=scatter!(βtest[:,1],βtest[:,2], group=(pval.<crit), legend=false,
               markersize=4, markerstrokewidth=0.0, seriesalpha=0.5,
               palette=:heat)
  b1 = lb[1]:(ub[1]-lb[1])/ngrid:ub[1]
  b2 = lb[2]:(ub[2]-lb[2])/ngrid:ub[2]
  colors = [:black, :red, :blue, :green]
  for t in 1:length(tests)
    fig=contour!(b1,b2,(a,b)->tests[t]([a,b]),
             levels = [0.9, 0.95],
             contour_labels=false, legend=false,
             label = labels[t],
             c=cgrad([colors[t],colors[t]],[0.0,1.0]))
  end
  fig
end


n = 50
k = 2
iv =3 
π0 = vcat(0.1*diagm(0=>ones(k)),0.2*ones(iv-k,k)) 
ρ = 0.5  
(y,x,z) = simulate_ivshare(n,β0,π0,ρ)
opt1 = optimize(θ->gmmObj(θ, β->gi_ivshare(β,y,x,z) ,I),
                β0, BFGS(), autodiff =:forward)
β1 = opt1.minimizer
V1 = gmmVar(β1,β->gi_ivshare(β,y,x,z),I)
  

pklm = θ->cdf(Chisq(length(βcue)),klm(θ, β->gi_ivshare(β,y,x,z)))
par  = θ->cdf(Chisq(size(z)[2]), ar(θ, β->gi_ivshare(β,y,x,z)))
pclr  = θ->clr(θ, β->gi_ivshare(β,y,x,z))
pwald = θ -> cdf(Chisq(length(β1)),(θ-β1)'*inv(V1)*(θ-β1))
plot_cr(β1,V1, [pclr, pklm, par, pwald],
        ["CLR","KLM","AR","Wald"], ngrid=40)