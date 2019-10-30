n = 2
psi = qAnneal.randomState(n)
psiB = qAnneal.cannonical_state(n,psi,10)
sig = qAnneal.decoherence(psiB, [1,1])
T,s = qAnneal.anneal(n,2,1,0,psiB)
if (@isdefined T) && (@isdefined a)
  return 0
end