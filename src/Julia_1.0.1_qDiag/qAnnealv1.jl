
module qAnneal

#import Gadfly for plots
#using Gadfly

using DelimitedFiles
using LinearAlgebra





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
function init(nQ, initWaveFun=missing)
    #declare all the variables and assign to global
    global n = nQ          #number of qubits of spinSystem

    #calculate the number of states and create the wave function
    global nStates = 2^n     #total number of states
    global waveFun = zeros(Complex{Float64}, nStates)  # wave function
    if initWaveFun === missing
      #fill!(waveFun,nStates^(-0.5))    # initializing the wave function to all x
      fill!(waveFun,1)
    else
      waveFun = convert(Array{Complex{Float64},1},initWaveFun)
    end
    #waveFun =  waveFun/norm(waveFun)
    normWaveFun()

    #global waveFunInterim = zeros(Complex{Float64}, nStates) #space to hold interim wave function

    #create the initial h and J matrix based on number of qubits
    global hxStart = [1 0 0]
    global hyStart = [1 0 0]
    global hzStart = [1 1 0]
    #global hxStart = zeros(1,n)

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
  waveFunInterim = zeros(Complex{Float64}, nStates)
  global Hop = zeros(Complex{Float64}, nStates, nStates)
  for k = 0:n-1
    i1=2^k
    hxi = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
    hyi = (1-delta)*hyStart[k+1] + delta*hy[k+1]
    hzi = (1-delta)*hzStart[k+1] + delta*hz[k+1]
    #based on equation 68
    hi = sqrt(hxi*hxi + hyi*hyi + hzi*hzi)
    if hi !=0
      sinTerm = sin(3*dt*hi/2)
      cosTerm = cos(3*dt*hi/2)
      spinOp11 = cosTerm + sinTerm*hzi*im/hi
      spinOp12 = (hxi*im+hyi)*sinTerm/hi
      spinOp21 = (hxi*im-hyi)*sinTerm/hi
      spinOp22 = cosTerm - sinTerm*hzi*im/hi
    else
      spinOp11 = 1;
      spinOp12 = 0;
      spinOp21 = 0;
      spinOp22 = 1;
    end
    for l=0:2:nStates-1
        i2= l & i1;
        i::Int = l - i2 + i2/i1 + 1;  #The +1 is because indexing in julia is 1-based
        j::Int = i + i1;
        #print("====",k," ",i," ",j," ",spinOp11," ",spinOp22, "\n")

        waveFunInterim[i] += spinOp11*waveFun[i] + spinOp12*waveFun[j]
        waveFunInterim[j] += spinOp21*waveFun[i] + spinOp22*waveFun[j]
        Hop[i,i] += spinOp11
        Hop[i,j] += spinOp12
        Hop[j,i] += spinOp21
        Hop[j,j] += spinOp22
        #print("===",i,":===",waveFunInterim[i],"===",j,":===",waveFunInterim[j],"\n")
        #print(spinOp11,",",spinOp12,",",waveFun[i],",",spinOp21,",",spinOp22,",",waveFun[j],"\n")
    end
  end
  #display(waveFun)
  #display("*1**")
  #display(Hop)
  #display(waveFunInterim) #/norm(waveFunInterim))
  #display("*2**")
  #display(Hop*waveFun) #/norm(Hop*waveFun))
  #display("*3**")

  for i::Int = 1:nStates
    global waveFun[i] = waveFunInterim[i]
  end

  normWaveFun()
  print("\n\n")


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
  waveFunInterim = zeros(Complex{Float64}, nStates)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1
        nii::Int=2^k
        njj::Int=2^l
        jxij=delta*jx[k+1,l+1]
        jyij=delta*jy[k+1,l+1]
        jzij=delta*jz[k+1,l+1]

        # equation 70
        a = 3*jzij                #somehow /4 is not coded in reference program
        b = 3*(jxij - jyij)
        c = 3*(jxij + jyij)

        dsOp11 = (exp(a*dt*im))*cos(b*dt)
        dsOp14 = (im*exp(a*dt*im))*sin(b*dt)
        dsOp22 = (exp(-a*dt*im))*cos(c*dt)
        dsOp23 = (im*exp(-a*dt*im))*sin(c*dt)
        dsOp32 = dsOp23
        dsOp33 = dsOp22
        dsOp41 = dsOp14
        dsOp44 = dsOp11

        #print(dsOp11,",",dsOp14,",",dsOp22,",",dsOp23,"\n")

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
      global  waveFun[i] = waveFunInterim[i]
    end

    normWaveFun()
end

"""
    normWaveFun()
Normalize the wave function.
# Arguments
* None
"""
function normWaveFun()
  normWaveFun = norm(waveFun)
  if normWaveFun == 0
    display("norm of wavefunction is zero. Something not right")
  else
    global waveFun = waveFun/normWaveFun
  end
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
    waveFunOp = zeros(Complex{Float64}, nStates)
    # Calculate (Σ hi σi)|Ψ>
    for k = 0:n-1
      i1 = 2^k
      hxi = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
      hyi = (1-delta)*hyStart[k+1] + delta*hy[k+1]
      hzi = (1-delta)*hzStart[k+1] + delta*hz[k+1]
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
            waveFunOp[n1] += -jzij*waveFun[n1] + jxij*waveFun[n2] + jyij*waveFun[n2]
            waveFunOp[n2] += -jzij*waveFun[n2] + jxij*waveFun[n1] + jyij*waveFun[n1]
            waveFunOp[n3] += jzij*waveFun[n3] + jxij*waveFun[n0] - jyij*waveFun[n0]
        end
      end
    end


    #find the system energy after the hamiltonina has been operated on
    #wavefunction. i.e find <Ψ|H|Ψ> given H|Ψ> from above.
    energy = sum(conj(waveFun).*waveFunOp)
end



function Hamiltonian(delta)

    # variable to hold H
    H = zeros(Complex{Float64}, nStates, nStates)
    # Calculate (Σ hi σi)
    for k = 0:n-1
      i1 = 2^k
      hxi = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
      hyi = (1-delta)*hyStart[k+1] + delta*hy[k+1]
      hzi = (1-delta)*hzStart[k+1] + delta*hz[k+1]
      for l=0:2:nStates-1
        i2= l & i1;
        i::Int = l - i2 +i2/i1 +1;   # The +1 is because indexing in julia is 1-based
        j::Int = i+i1;
        #waveFunOp[i] += (hxi - hyi*im)*waveFun[j] + hzi*waveFun[i]
        #waveFunOp[j] += (hxi + hyi*im)*waveFun[i] - hzi*waveFun[j]
        H[i,j] += hxi - hyi*im
        H[i,i] += hzi
        H[j,i] += hxi + hyi*im
        H[j,j] += - hzi
      end
    end

    # Calculate (Σ Jij σi σj)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1
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
            #waveFunOp[n0] += jzij*waveFun[n0] + jxij*waveFun[n3] - jyij*waveFun[n3]
            #waveFunOp[n1] += jzij*waveFun[n1] + jxij*waveFun[n2] + jyij*waveFun[n2]
            #waveFunOp[n2] += jzij*waveFun[n2] + jxij*waveFun[n1] + jyij*waveFun[n1]
            #waveFunOp[n3] += jzij*waveFun[n3] + jxij*waveFun[n0] - jyij*waveFun[n0]
            H[n0,n0] += jzij
            H[n0,n3] += jxij - jyij
            H[n1,n1] += -jzij
            H[n1,n2] += jxij + jyij
            H[n2,n2] += -jzij
            H[n2,n1] += jxij + jyij
            H[n3,n3] += jzij
            H[n3,n0] += jxij - jyij
        end
      end
    end


    return -H

end

"""
Calculate s from t.
currently, it is linear. Change this function for more complex evolution.
"""
function get_s(t)
  return t    #s is linear in t,
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
function anneal(nQ, t=1.0, dt=0.1, initWaveFun=missing)
    println("Annealing started...")

    #initialize the system
    init(nQ, initWaveFun)

    #global energy = zeros(Complex{Float64}, nSteps+1)
    #nSteps = t/dt      #number of steps

    #loops for each increment in time and calculates the new wavefunction.
    #for i=0:nSteps
    for time_step = 0:dt:t-dt
        #delta = i/nSteps
        display(time_step)
        display(waveFun)
        s = get_s(time_step/t)
        #print(s," ",dt, "\n ")
        singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        doubleSpinOp(s, dt)
        #doubleSpinOp(s, dt)
        #doubleSpinOp(s, dt)
        singleSpinOp(s, dt)
        display("*******************************")
        #energy[i+1]=energySys(delta)
    end

    println("Annealing done...")
    return waveFun

    #plot(x=0:nSteps,y=real(energy),Geom.line)

end

end

"""
qAnneal.init(3)
qAnneal.anneal(3,1.0,1/19)

using LinearAlgebra
H_init=qAnneal.Hamiltonian(0)
H_fin=qAnneal.Hamiltonian(1)
psi0=ones(8)
psi0 /= norm(psi0)
qDiag.diag_evolution(H_init, H_fin, psi0, 1, 0.0005)
"""
