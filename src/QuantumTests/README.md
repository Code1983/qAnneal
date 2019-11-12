#calculates quantum decoherence with time

The multi-core julia code for quantum annealing and decoherence calculations.
A sample code to run from Julia repl is present below.

```
using Distributed
using SharedArrays

addprocs(1)
@everywhere include("qAnneal.jl")
@everywhere qAnneal.getConfig("../config_4_16_decoh")
n=4    # Number of Qubits.

# The below coide is for Annealing
b=qAnneal.anneal(n,5,0.1)

# The below code is for calculating decoherence over time.
psi = qAnneal.randomState(n)
psiB = qAnneal.cannonical_state(n,psi,10)
sig = qAnneal.decoherence(psiB, [2,2])
T,s = qAnneal.annealTherm(n,2,1,[2, 2], 0,psiB)
```
