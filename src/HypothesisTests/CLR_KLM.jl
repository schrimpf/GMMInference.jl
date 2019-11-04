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