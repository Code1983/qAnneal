include("qAnneal.jl")
qAnneal.getConfig("../config_4_16_decoh")
n = 4
psi = qAnneal.randomState(n)
psiB = qAnneal.cannonical_state(n,psi,10)
sig = qAnneal.decoherence(psiB, [2,2])
T,s = qAnneal.annealTherm(n,2,1,[2, 2], 0,psiB)
if (@isdefined T) && (@isdefined a)
  return 0
end
