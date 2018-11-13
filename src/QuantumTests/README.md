# qAnneal

Quantum Annealing in Julia. 
This has three functions for annealing
* `anneal(nQ, t=1.0, dt=0.1, initWaveFun=missing)` uses Sujuki-Trotter product-formula algorithm. nQ is number of qubits.
* `diag_evolve(t, dt)` uses diagonalization for evolution to the exact initial conditions as was used in previous function.
* `diag_evolution(H_init, H_fin, wavefun, t, dt)` also uses diagonalization but can have any initial and final hamiltonian.


# Short overview on usage

Update the h and J files in the folder where code is present. Run qAnnealv1.jl and then 

```
a=qAnneal.anneal(9,1,0.0005)   # The H* and J* files has the bias and coupling. 
                               # 9 is the number of qubits. 
                               # 1.0 is the time of evolution 0.0005 is dt(time-step) of evolution.
                               # The The fuction returns the final waveFunction. It should be an array of (2^9 = )512 

b=qAnneal.diag_evolve(1,0.0005) # Evolves the same initial condition as the previous code 
                                # but uses diagonalization instead of suzuki trotter.
                                # 1.0 is the time of evolution 0.0005 is dt(time-step) of evolution. 
                                # Please run anneal() before you run diag_evolve()
```

The instructions are detailed in the docs folder.
