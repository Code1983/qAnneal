
module qAnneal

#import Gadfly for plots
using Gadfly





"""
    init(n)
Initializes the spin system
# Arguments
* `n::Integer`: number of cubits of system.
# What it does
* Creates the initial wavefunction with all cubits in x-direction
* creates the initial h-matrix which aligns cubits to x-direction
* creates global variables for use in this package
"""
function init(nQ)
    #declare all the variables and assign to global
    global n = nQ          #number of qubits of spinSystem

    #calculate the number of states and create the wave function
    global nStates = 2^n     #total number of states
    global waveFun = zeros(Complex128, nStates)  # wave function
    fill!(waveFun,nStates^(-0.5))    # initializing the wave function to all x
    global waveFunInterim = zeros(Complex128, nStates) #space to hold interim wave function

    #create the initial h and J matrix based on number of qubits
    global hxStart = ones(1,n)

    #read the J and h files and create corresponding matrices
    #This corresponds to the final influences to the system.
    global jz = readdlm("./Jz.txt")
    global jx = readdlm("./Jx.txt")
    global jy = readdlm("./Jy.txt")
    global hz = readdlm("./hz.txt")
    global hx = readdlm("./hx.txt")
    global hy = readdlm("./hy.txt")

    #print the array, plot the matrix etc
    println("Initialization done...")
end

"""
    singleSpinOp(delta, dt)
Single spin operator for time evolution. Based on eqn-68 from the paper.
This funtion and doubleSpinOp updates the wavefunction.
# Arguments
* `delta`: Time since initial configuration.
* `dt`: duration of time steps.
"""
function singleSpinOp(delta, dt)
  global waveFunInterim = zeros(Complex128, nStates)
  for k = 0:n-1
    i1=2^k
    hxi = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
    hyi = delta*hy[k+1]
    hzi = delta*hz[k+1]
    #based on equation 68
    hi = sqrt(hxi*hxi + hyi*hyi + hzi*hzi)
    if hi !=0
      sinTerm = sin(dt*hi/2)
      cosTerm = cos(dt*hi/2)
      spinOp11 = cosTerm + sinTerm*hzi*im/hi
      spinOp12 = (hyi+hxi*im)*sinTerm/hi
      spinOp21 = (hyi-hxi*im)*sinTerm/hi
      spinOp22 = cosTerm - sinTerm*hzi*im/hi
      for l=0:2:nStates-1
        i2= l & i1;
        i::Int = l - i2 + i2/i1 + 1;  #The +1 is because indexing in julia is 1-based
        j::Int = i + i1;

        waveFunInterim[i] += spinOp11*waveFun[i] + spinOp12*waveFun[j]
        waveFunInterim[j] += spinOp21*waveFun[i] + spinOp22*waveFun[j]
      end
    end
  end
end

"""
    doubleSpinOp(delta, dt)
Double spin operator for time evolution. Based on eqn-70 from paper.
This funtion and singleSpinOp updates the wavefunction.
# Arguments
* `delta`: Time since initial configuration step.
* `dt`: duration of time steps.
"""
function doubleSpinOp(delta, dt)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1    # Do not understand why only upper matrix is considered.
        nii::Int=2^k
        njj::Int=2^l
        jxij=delta*jx[k+1,l+1]
        jyij=delta*jy[k+1,l+1]
        jzij=delta*jz[k+1,l+1]

        # equation 70
        a = jzij/4
        b = (jxij - jyij)/4
        c = (jxij + jyij)/4

        dsOp11 = (e^(a*dt*im))*cos(b*dt)
        dsOp14 = (im*e^(a*dt*im))*sin(b*dt)
        dsOp22 = (e^(-a*dt*im))*cos(c*dt)
        dsOp23 = (im*e^(-a*dt*im))*sin(c*dt)
        dsOp32 = dsOp23
        dsOp33 = dsOp22
        dsOp41 = dsOp14
        dsOp44 = dsOp11

        for m::Int = 0:4:nStates-1
            n3::Int = m & njj;
            n2::Int = m-n3+(n3+n3)/njj;
            n1::Int = n2 & nii;
            n0::Int = n2 - n1+n1/nii +1; # The +1 is because indexing in julia is 1-based
            n1=n0+nii;
            n2=n0+njj;
            n3=n1+njj;
            waveFunInterim[n0] += dsOp11*waveFun[n0] + dsOp14*waveFun[n3]
            waveFunInterim[n1] += dsOp22*waveFun[n1] + dsOp23*waveFun[n2]
            waveFunInterim[n2] += dsOp32*waveFun[n1] + dsOp33*waveFun[n2]
            waveFunInterim[n3] += dsOp41*waveFun[n0] + dsOp44*waveFun[n3]
        end
      end
    end

    #move the interim values of wavefunction to existing wavefunction.
    for i::Int = 1:nStates
        waveFun[i] = waveFunInterim[i]
    end
end

"""
    normWaveFun()
Normalize the wave function.
# Arguments
* None
"""
function normWaveFun()
    waveNorm = norm(qAnneal.waveFun)
    global waveFun /= waveNorm
end

"""
    energySys(n)
calculates the global energy of the system
# Arguments
* `n::Integer`: number of cubits of system.
# What it does
We are calculating <Ψ|H|Ψ> where H is the hamiltonian given by
H = Σ Jij σi σj - Σ hi σi  (equation 20)
Following is the sequence of steps
 1. First we calculate H|Ψ>. This is broadly done in two steps
        1a. calculate (Σ hi σi)|Ψ>
        1b. calculate (Σ Jij σi σj)|Ψ>
 2. Finally calculate energy by mutiplying <Ψ| with H|Ψ> from previous step
"""
function energySys(delta)

    # variable to hold H|Ψ>
    waveFunOp = zeros(Complex128, nStates)
    # Calculate (Σ hi σi)|Ψ>
    for k = 0:n-1
      i1 = 2^k
      hxi = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
      hyi = delta*hy[k+1]
      hzi = delta*hz[k+1]
      for l=0:2:nStates-1
        i2= l & i1;
        i::Int = l - i2 +i2/i1 +1;   # The +1 is because indexing in julia is 1-based
        j::Int = i+i1;
        waveFunOp[i] += (hxi - hyi*im)*waveFun[j] + hzi*waveFun[i]
        waveFunOp[j] += (hxi + hyi*im)*waveFun[i] - hzi*waveFun[j]
      end
    end

    # Calculate (Σ Jij σi σj)|Ψ>
    for k::Int = 0:n-1
      for l::Int = k+1:n-1    # Do not understand why only upper matrix is considered.
        nii::Int=2^k
        njj::Int=2^l
        jxij=delta*jx[k+1,l+1]
        jyij=delta*jy[k+1,l+1]
        jzij=delta*jz[k+1,l+1]
        for m::Int = 0:4:nStates-1
            n3::Int = m & njj;
            n2::Int = m-n3+(n3+n3)/njj;
            n1::Int = n2 & nii;
            n0::Int = n2 - n1+n1/nii +1; # The +1 is because indexing in julia is 1-based
            n1=n0+nii;
            n2=n0+njj;
            n3=n1+njj;
            # current the code is only of Jz*Jz only
            waveFunOp[n0] += jzij*waveFun[n0] + jxij*waveFun[n3] - jyij*waveFun[n3]
            waveFunOp[n1] += jzij*waveFun[n1] + jxij*waveFun[n2] + jyij*waveFun[n2]
            waveFunOp[n2] += jzij*waveFun[n2] + jxij*waveFun[n1] + jyij*waveFun[n1]
            waveFunOp[n3] += jzij*waveFun[n3] + jxij*waveFun[n0] - jyij*waveFun[n0]
        end
      end
    end


    #find the system energy after the hamiltonina has been operated on
    #wavefunction. i.e find <Ψ|H|Ψ> given H|Ψ> from above.
    energy = sum(conj(waveFun).*waveFunOp)

end

"""
    Anneal(nQ, nSteps=10)
Anneals the spin system. This is the evolution of spin system from a known
initial state to final state. The initial state is all cubits with spin in
x-direction. The final state is specified in the h and J files.
The algorithm is based on Sujuki-Trotter product formula described in
`H. De Raedt and K. Michielsen. Computational Methods for Simulating Quantum
Computers. In M. Rieth and W. Schommers, editors, Handbook of Theoretical and
Computational Nanotechnology, volume 3, chapter 1, page 248. American Scientific
Publisher, Los Angeles, 2006.`
# Arguments
* `nQ::Integer`: number of cubits of system.
* `nSteps`: number to time steps in which we reach the end system configuration.
# Usage
```Julia
julia> qAnneal.anneal(3, 10)
```
"""
function anneal(nQ, nSteps=10)
    println("Annealing started...")

    #initialize the system
    init(nQ)
    global energy = zeros(Complex128, nSteps+1)
    dt = 1.0/nSteps      #time steps

    #loops for each increment in time and calculates the new wavefunction.
    for i=0:nSteps
        delta = i/nSteps
        singleSpinOp(delta, dt)
        doubleSpinOp(delta, dt)
        normWaveFun()
        energy[i+1]=energySys(delta)
    end
    println("Annealing done...")

    plot(x=0:nSteps,y=real(energy),Geom.line)

end

end
