#calculates quantum decoherence with time

A sample code to run from Julia repl is present below.

```
@everywhere include("qAnnealv6.jl")
n = 20
psi = qAnneal.randomState(n)
psiB = qAnneal.cannonical_state(n,psi,10,100)
sig = qAnneal.decoherence(psiB, [4,16])
qAnneal.anneal(n,2,1,0,psiB)
qAnneal.anneal(n,100,5,0,psiB)
```
