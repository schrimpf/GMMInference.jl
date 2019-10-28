"""
    MixturePanel <: GMMModel

A `MixturePanel` model consists of outcomes, `y`, regressors, `x`, and
a number of types, `ntypes`. Each observation `i` is one of `ntypes`
types with probability `p[i]`.  Conditional on type,`y` is given by

` y[i,t] = x[i,t,:]*β[:,type[i]] + ϵ[i,t] `

It is assumed that `ϵ` is uncorrelated accross `i` and `t` and E[ϵ²]= σ².

The moment conditions used to estimate `p`, `β` and `σ` are

``E[ \\sum_{j} x(y - xβ[:,j])p[j]] = 0 ``

and 

``E[ y[i,t]*y[i,s] - \\sum_j p[j](x[i,t,:]β[:,j] x[i,s,:]β[:,j]) -1(t=s)σ²] = 0``

"""
struct MixturePanel <: GMMModel
  x::Array{Float64,3}
  y::Matrix{Float64}
  ntype::Integer
end

"""
    MixturePanel(n::Integer, t::Integer,
                      k::Integer, type_prob::AbstractVector,
                      β::AbstractMatrix, σ = 1.0)

Simulate a MixturePanel model.

# Arguments

- `n` individuals
- `t` time periods
- `k` regressors
- `type_prob` probability of each type
- `β` matrix `(k × lenght(type_prob))` coefficients
- `σ` standard deviation of `ϵ`

Returns a MixturePanel GMMModel.
"""
function MixturePanel(n::Integer, t::Integer,
                      k::Integer, type_prob::AbstractVector,
                      β::AbstractMatrix, σ = 1.0)
  x = randn(n, t, k)
  ntype=length(type_prob)
  type = rand(Categorical(type_prob), n)
  y = Matrix(undef, n, t)
  for i in 1:n
    y[i,:] = x[i,:,:] * β[:,type[i]] + randn(t)*σ
  end
  MixturePanel(x,y,ntype)
end

number_parameters(model::MixturePanel) = size(model.x,3)*model.ntype+model.ntype
number_observations(model::MixturePanel) = size(model.y,1)
number_moments(model::MixturePanel) = (size(model.y,2)*size(model.x,3) +
                                       sum(1:size(model.y,2)) )

function get_gi(model::MixturePanel)
  function(θ)
    N = size(model.y,1)
    T = size(model.y,2)
    K = size(model.x,3)
    types = model.ntype
    p = zeros(eltype(θ), types)
    p[1:(types-1)] = θ[1:(types-1)]
    p .= exp.(p)
    p .= p./sum(p)
    σ = exp(θ[types])
    β = reshape(θ[(types+1):end], K, types)
    moments = zeros(eltype(θ), N, number_moments(model))
    @inbounds for i in 1:N
      xb = model.x[i,:,:]*β
      et = model.y[i,:] - xb*p
      for k in 1:K
        moments[i,((k-1)*T+1):(k*T)] .= et.*model.x[i,:,k]
      end
      m = K*T+1
      for t in 1:T
        for s in t:T
          moments[i,m] = model.y[i,t]*model.y[i,s]- dot(xb[t,:].*xb[s,:],p) - (s==t)*σ^2
          m+=1
        end
      end
    end
    return(moments)
  end
end

