# GMMInference.jl


## GMMModel

The abstract type `GMMModel` is used to define methods for generic
method of moments problems. To use this interface, define a concrete 
subtype of `GMMModel` and at least a specialized `get_gi` method. See
`src/models` for a few examples. 

```@autodocs
Modules = [GMMInference]
Pages = ["GMMInference.jl"]
```

```@autodocs
Modules = [GMMInference]
Pages = ["gmmmodel.jl"]
```

### IV Logit

```@autodocs
Modules = [GMMInference]
Pages = ["ivlogitshare.jl"]
```

### Panel Mixture

```@autodocs
Modules = [GMMInference]
Pages = ["mixturepanel.jl"]
```

### Random Coefficients Logit

```@autodocs
Modules = [GMMInference]
Pages = ["rclogit.jl"]
```

## Inference

Inference methods for GMM problems. Currently only contains GEL based
tests. See [empirical
likelihood](https://schrimpf.github.io/GMMInference.jl/empiricalLikelihood/)
for usage, background, and references.


```@autodocs
Modules = [GMMInference]
Pages = ["inference.jl"]
```

We plan to also add AR, KLM, and CLR methods for GMM. See [identification
robust
inference](https://schrimpf.github.io/GMMInference.jl/identificationRobustInference/). 
