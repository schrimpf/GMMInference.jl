"""
    IVLogitShare <: GMMModel

An `IVLogitShare` model consists of outcomes, `y` ∈ (0,1),  regressors
`x` and instruments `z`.  The moment condition is

``E[ (\\log(y/(1-y)) - xβ)z ] = 0``

The dimensions of `x`, `y`, and `z` must be such that
`length(y) == size(x,1) == size(z,1)`
and 
`size(x,2) ≤ size(z,2)`.
"""
struct IVLogitShare <: GMMModel
  x::Matrix{Float64}
  y::Vector{Float64}
  z::Matrix{Float64}
end

"""
    IVLogitShare(n::Integer, β::AbstractVector,
                      π::AbstractMatrix, ρ)

Simulate an IVLogitShare model. 

# Arguments

- `n` number of observations
- `β` coefficients on `x`
- `π` first stage coefficients `x = z*π + v`
- `ρ` correlation between x[:,1] and structural error.

Returns an IVLogitShare GMMModel.
"""
function IVLogitShare(n::Integer, β::AbstractVector,
                      π::AbstractMatrix, ρ)
  z = randn(n, size(π)[1])
  endo = randn(n, length(β))
  x = z*π .+ endo
  ξ = rand(Normal(0,sqrt((1.0-ρ^2))),n).+endo[:,1]*ρ
  y = cdf.(Logistic(), x*β .+ ξ)
  return(IVLogitShare(x,y,z))
end

number_parameters(model::IVLogitShare) = size(model.x,2)
number_observations(model::IVLogitShare) = length(model.y)
number_moments(model::IVLogitShare) = size(model.z,2)

function get_gi(model::IVLogitShare)
  function(β)
    ξ = quantile.(Logistic(), model.y) .- model.x*β
    ξ.*model.z
  end
end

function gel_jump_problem(model::IVLogitShare)
  n = number_observations(model)
  d = number_parameters(model)
  k = number_moments(model)
  Ty = quantile.(Logistic(),model.y)   
  m = Model()
  @variable(m, 0.0 <= p[1:n] <= 1.0)
  @variable(m, θ[1:d])
  @constraint(m, prob,sum(p)==1.0)
  @constraint(m, momentcon[i=1:k], dot((Ty - model.x*θ).*model.z[:,i],p)==0.0)
  @NLobjective(m,Max, sum(log(p[i]) for i in 1:n))
  return(m)
end

