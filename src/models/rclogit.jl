"""
    RCLogit <: GMMModel

A random coefficients logit model with endogeneity. 
An `RCLogit` model consists of outcomes, `y` ∈ (0,1),  regressors
`x`, instruments `z`, and random draws `ν ∼ N(0,I)`.  The moment condition is

``E[ξz] = 0``

where 

`` y = ∫ exp(x(β + ν) + ξ)/(1 + exp(x(β + ν) + ξ)) dΦ(ν;Σ) ``

where Φ(ν;Σ) is the normal distribution with variance Σ.

The dimensions of `x`, `y`, `z`, and `ν` must be such that
`length(y) == size(x,1) == size(z,1) == size(ν,2)`
and
`size(ν,3) == size(x,2) ≤ size(z,2)`. 
"""
struct RCLogit <: GMMModel
  x::Matrix{Float64}
  y::Vector{Float64}
  z::Matrix{Float64}
  ν::Array{Float64,3}
end

"""
    RCLogit(n::Integer, β::AbstractVector,
            π::AbstractMatrix, Σ::AbstractMatrix,
            ρ, nsim=100)

Simulates a RCLogit model.

# Arguments

- `n` number of observations
- `β` mean coefficients on `x`
- `π` first stage coefficients `x = z*π + e`
- `Σ` variance of random coefficients
- `ρ` correlation between x[:,1] and structural error.
- `nsim` number of draws of `ν` for monte-carlo integration
"""
function RCLogit(n::Integer, β::AbstractVector,
                 π::AbstractMatrix, Σ::AbstractMatrix,
                 ρ, nsim=100)
  z = randn(n, size(π)[1])
  endo = randn(n, length(β))
  x = z*π .+ endo
  ξ = rand(Normal(0,sqrt((1.0-ρ^2))),n).+endo[:,1]*ρ
  ν = randn(nsim,n,length(β))
  U = cholesky(Σ).U
  y = zeros(n)
  for s in 1:nsim
    y .= y .+ cdf.(Logistic(), x*β .+ sum(x.*(ν[s,:,:]*U),dims=2) .+ ξ)[:]
  end
  y  .= y./nsim
  #ν = randn(nsim÷10,n,length(β))
  RCLogit(x,y,z,ν)
end

number_parameters(model::RCLogit) = size(model.x,2) +
  size(model.x,2)*(size(model.x,2)-1) ÷ 2 + number_observations(model)
number_observations(model::RCLogit) = length(model.y)
number_moments(model::RCLogit) = size(model.z,2)

function get_gi(model::RCLogit)
  function(θ)
    n = number_observations(model)
    ξ = θ[(length(θ)-n+1):end]
    ξ.*model.z
  end
end

"""
    gmm_constraints(model::RCLogit)

Returns 

`` c(θ) = ∫ exp(x(β + ν) + ξ)/(1 + exp(x(β + ν) + ξ)) dΦ(ν;Σ) - y``

where `θ = [β, uvec, ξ]` with `uvec = vector(cholesky(Σ).U)`.

The integral is computed by monte-carlo integration. 
"""
function gmm_constraints(model::RCLogit)
  function(θ)
    (nsim,n,K) = size(model.ν)
    β = θ[1:K]
    uvec = θ[(K+1):(K+(K*(K+1))÷2)]
    U =  zeros(eltype(uvec),K,K)
    l = 1
    for i in 1:K
      for j in i:K
        U[i,j] = uvec[l]
        l = l+1
      end
    end
    ξ = θ[((K+(K*(K-1))÷2)+1):end]
    share = zeros(eltype(θ), n)
    for s in 1:nsim      
      share += cdf.(Logistic(), model.x*β + sum(model.x.*(model.ν[s,:,:]*U),dims=2) .+ ξ)[:]
    end
    share /= nsim
    return(share - model.y)
  end
end

