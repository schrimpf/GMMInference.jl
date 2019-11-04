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
    ivlogitshare = IVLogitShare(n,β0,π0,ρ)
    gi = get_gi(ivlogitshare)
    return AR_gmm_obj(gi)
end




n = 100
k = 2
iv = 3
β0 = ones(k)
π0 = vcat(5*I,ones(iv-k,k)) 
ρ = 0.5  
ar_ivshare = AR_test(n,β0,π0,ρ)


optcue = optimize(ar_ivshare,
                  β0, BFGS(), autodiff =:forward)
@show βcue = optcue.minimizer