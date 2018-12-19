
module qAnneal

#import Gadfly for plots
#using Gadfly

using DelimitedFiles
using LinearAlgebra

#create the initial h and J matrix based on number of qubits
const hxStart = convert(Array{Float32,2},readdlm("./hx_init.txt"))
const hyStart = convert(Array{Float32,2},readdlm("./hy_init.txt"))
const hzStart = convert(Array{Float32,2},readdlm("./hz_init.txt"))
#read the J and h files and create corresponding matrices
#This corresponds to the final influences to the system.const jz = readdlm("./Jz.txt")
const jz = readdlm("./Jz.txt")
const jx = readdlm("./Jx.txt")
const jy = readdlm("./Jy.txt")
const hz = readdlm("./hz.txt")
const hx = readdlm("./hx.txt")
const hy = readdlm("./hy.txt")


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
function init(nStates::Int, initWaveFun=missing)

  global waveFun = zeros(Complex{Float32}, nStates)  # wave function
  if initWaveFun === missing
    #fill!(waveFun,nStates^(-0.5))    # initializing the wave function to all x
    fill!(waveFun,1)
  else
    waveFun = convert(Array{Complex{Float32},1},initWaveFun)
  end
  waveFun = normWaveFun(waveFun)
  println("Initialization done...")
  return waveFun
end

"""
    singleSpinOp(delta, dt)
Single spin operator for time evolution. Based on eqn-68 from the paper.
This funtion and doubleSpinOp updates the wavefunction.
# Arguments
* `delta`: Time since initial configuration.
* `dt`: duration of time steps.
"""
function singleSpinOp(n, nStates, delta, dt, waveFun)
  waveFunInterim = zeros(Complex{Float32}, nStates)
  for k = 0:n-1
    i1=2^k
    hxi = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
    hyi = (1-delta)*hyStart[k+1] + delta*hy[k+1]
    hzi = (1-delta)*hzStart[k+1] + delta*hz[k+1]
    #based on equation 68
    hi = sqrt(hxi*hxi + hyi*hyi + hzi*hzi)
    if hi !=0
      sinTerm = sin(n*dt*hi/2)
      cosTerm = cos(n*dt*hi/2)
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
    Threads.@threads for l=0:2:nStates-1
        i2= l & i1;
        i::Int = l - i2 + i2/i1 + 1;  #The +1 is because indexing in julia is 1-based
        j::Int = i + i1;
        #print("====",k," ",i," ",j," ",spinOp11," ",spinOp22, "\n")

        waveFunInterim[i] += spinOp11*waveFun[i] + spinOp12*waveFun[j]
        waveFunInterim[j] += spinOp21*waveFun[i] + spinOp22*waveFun[j]
        #Hop[i,i] += spinOp11
        #Hop[i,j] += spinOp12
        #Hop[j,i] += spinOp21
        #Hop[j,j] += spinOp22
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

  #Threads.@threads for i::Int = 1:nStates
  #  global waveFun[i] = waveFunInterim[i]
  #end
  #global waveFun = waveFunInterim

  return normWaveFun(waveFunInterim)


end




"""
    doubleSpinOp(delta, dt)
Double spin operator for time evolution. Based on eqn-70 from paper.
This funtion and singleSpinOp updates the wavefunction.
# Arguments
* `delta`: Time since initial configuration step.
* `dt`: duration of time steps.
"""
function doubleSpinOp(n, nStates, delta, dt, waveFun)
  waveFunInterim = zeros(Complex{Float32}, nStates)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1
        nii::Int=2^k
        njj::Int=2^l
        jxij=delta*jx[k+1,l+1]
        jyij=delta*jy[k+1,l+1]
        jzij=delta*jz[k+1,l+1]

        # equation 70
        a = jzij                #somehow /4 is not coded in reference program
        b = (jxij - jyij)
        c = (jxij + jyij)

        ndt = n*(n-1)*dt/2

        dsOp11 = (exp(a*ndt*im))*cos(b*ndt)
        dsOp14 = (im*exp(a*ndt*im))*sin(b*ndt)
        dsOp22 = (exp(-a*ndt*im))*cos(c*ndt)
        dsOp23 = (im*exp(-a*ndt*im))*sin(c*ndt)
        dsOp32 = dsOp23
        dsOp33 = dsOp22
        dsOp41 = dsOp14
        dsOp44 = dsOp11

        #print(dsOp11,",",dsOp14,",",dsOp22,",",dsOp23,"\n")

        Threads.@threads for m::Int = 0:4:nStates-1
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
    #Threads.@threads for i::Int = 1:nStates
    #  global  waveFun[i] = waveFunInterim[i]
    #end
    #global  waveFun = waveFunInterim

    return normWaveFun(waveFunInterim)
end

"""
    normWaveFun()
Normalize the wave function.
# Arguments
* None
"""
function normWaveFun(waveFun)
  normWaveFun = norm(waveFun)
  if normWaveFun == 0
    display("norm of wavefunction is zero. Something not right")
  else
    return waveFun/normWaveFun
  end
end

"""
    energySys(n, wavefunction)
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
function energySys(n::Int, nStates::Int, delta, wavefun = missing)

    # variable to hold H|Ψ>
    waveFunOp = zeros(Complex{Float32}, nStates)
    if wavefun === missing
      wavefun = waveFun
    end
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
        waveFunOp[i] += (hxi - hyi*im)*wavefun[j] + hzi*wavefun[i]
        waveFunOp[j] += (hxi + hyi*im)*wavefun[i] - hzi*wavefun[j]
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
            waveFunOp[n0] += jzij*wavefun[n0] + jxij*wavefun[n3] - jyij*wavefun[n3]
            waveFunOp[n1] += -jzij*wavefun[n1] + jxij*wavefun[n2] + jyij*wavefun[n2]
            waveFunOp[n2] += -jzij*wavefun[n2] + jxij*wavefun[n1] + jyij*wavefun[n1]
            waveFunOp[n3] += jzij*wavefun[n3] + jxij*wavefun[n0] - jyij*wavefun[n0]
        end
      end
    end


    #find the system energy after the hamiltonina has been operated on
    #wavefunction. i.e find <Ψ|H|Ψ> given H|Ψ> from above.
    energy = -sum(conj(wavefun).*waveFunOp)
    return energy
end

function energyByMatrix(delta=1, H=missing, wavefun=missing)
  if H === missing && wavefun == missing
    energy = conj(waveFun)*Hamiltonian(delta)*waveFun
  else
    energy = conj(wavefun)*H*waveFun
  end
  return energy
end


"""
Returns state vector from Bloch Sphere notation.
"""
function blochSphere(theta , phi )
  cosTerm = cos(theta/2)
  sinTerm = exp(phi*im)*sin(theta/2)
  #print( "(",cosTerm , ")|0> + (" , sinTerm ,")|1>", "\n")
  return [cosTerm ; sinTerm]
end



function Hamiltonian(n, nStates, delta)

    # variable to hold H
    H = zeros(Complex{Float32}, nStates, nStates)
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
function anneal(nQ::Int, t=1.0, dt=0.1, initWaveFun=missing)
    println("Annealing started...")

    #initialize the system
    nStates::Int = 2^nQ     #total number of states
    waveFun = init(nStates, initWaveFun)

    for time_step = 0:dt:t-dt
        #delta = i/nSteps
        #display(time_step)
        #display(waveFun)
        s = get_s(time_step/t)
        #print(s," ",dt, "\n ")
        waveFun = singleSpinOp(nQ, nStates, s, dt, waveFun)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        waveFun = doubleSpinOp(nQ, nStates, s, dt, waveFun)
        #doubleSpinOp(s, dt)
        #doubleSpinOp(s, dt)
        waveFun = singleSpinOp(nQ, nStates, s, dt, waveFun)
        #display("*******************************")
        #energy[i+1]=energySys(delta)
    end

    println("Annealing done...")
    return waveFun

    #plot(x=0:nSteps,y=real(energy),Geom.line)

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


h_bar = 1
"""
Diagonalize the hamiltonian and evolve the wave function.
"""
function diag_dt(H, wavefun, dt)
  if (!ishermitian(H))
    display("H is not hermitian.")
    return 0
  end

  H_herm = Hermitian(H)

  eig = eigen(H_herm)
  H_eig_diag = Diagonal(eig.values)
  v = eig.vectors

  new_wavefun = v * exp(-im*dt*H_eig_diag/h_bar) * v' * wavefun
  return new_wavefun/norm(new_wavefun)
end


"""
Transition H from initial to final configuration.
"""
function get_h(H_init, H_fin, s)
  if (s>1)
    display("Incorrect S value.",S)
    return 0
  end
  H_new = (1-s)*H_init + s*H_fin
end

"""
evolve wavefuntion by changing H from initial to final configuration.
"""
function diag_evolution(H_init, H_fin, wavefun, t, dt)  #test with linear progress of time.
  steps = t/dt
  for time_step = 0:dt:t-dt
    s = get_s(time_step/t)
    H = get_h(H_init, H_fin, s)
    #wavefun = diag_dt(H, wavefun, dt)
    #display(time_step)
    #display(wavefun)
    #display(H)
    wavefun = diag_dt(H, wavefun, dt)
    #display("*******************************")
  end
  #display("*******************************")
  #display(time_step)
  #display(wavefun)
  return wavefun
end



"""
evolve wavefuntion by changing H from initial to final configuration.
"""
#test with linear progress of time.
function diag_evolve(nQ::Int, t=1.0, dt=0.1, initWaveFun=missing)
  #initialize the system
  nStates::Int = 2^nQ     #total number of states
  init(nStates, initWaveFun)
  steps = t/dt
  H_init=Hamiltonian(nQ, nStates, 0)
  H_fin=Hamiltonian(nQ, nStates, 1)
  #wavefun=copy(waveFun_init)
  for time_step = 0:dt:t-dt
    s = get_s(time_step/t)
    H = get_h(H_init, H_fin, s)
    #wavefun = diag_dt(H, wavefun, dt)
    #display(time_step)
    #display(wavefun)
    #display(H)
    waveFun = diag_dt(H, waveFun, dt)
    #display("*******************************")
  end
  #display("*******************************")
  #display(time_step)
  #display(wavefun)
  return waveFun
end

"""
convert a decimal number to a binary string.
"""
function decToBin(x::Int)

  if x%2 == 0
    bin = "0"
  else
    bin = "1"
  end

  x = floor(Int,x/2)
  while x > 0
    if x%2 == 0
      bin = "0"*bin
    else
      bin = "1"*bin
    end
    x = floor(Int,x/2)
  end
  return bin

end

"""
Reverse the order of binary bits from least significant bit representing firt
bit to most significant bit representing first bit.

for example, in a 3 bit series convert from
000 001 010 011 100 101 110 111
to
000 100 010 110 001 101 011 111
Then convert the binary number back to decimal
"""
function reverseBinDec(num, n)

  bin = decToBin(num)
  bin = lpad(bin,n,'0')
  i = 0
  reverseNum = 0
  for c in bin
    if c == '1'
      reverseNum += 2^i
    end
    i = i+1
  end
  return reverseNum

end

"""
convert wavefunction representation from least signifcant qubit to most
significant qubit. This code outputs results where least significant bit is
first qubit.
"""
function convertWavefunIndex(w,n)
  wNew = copy(w)
  for i = 0:2^n-1
    j = reverseBinDec(i, n)
    wNew[j+1]=w[i+1]
  end
  return wNew
end

function convertHamiltonianIndex(H,n)
  Hnew = copy(H)
  for i = 0:2^n-1
    for j = 0:2^n-1
      k = reverseBinDec(i, n)
      l = reverseBinDec(j, n)
      #print(i,' ',j,' ',k,' ',l,"\n")
      Hnew[k+1, l+1]=H[i+1, j+1]
    end
  end
  return Hnew
end

"""
wavefunction from a given state of qubit.
The least significant bit is qubit 1.

Use convertWavefunIndex() if you are following the other convention of most
significant bit as first qubit.
"""
function qubitsToWavefun(qState)
  n = length(qState)
  index = 0
  for i=1:n
    index += qState[i]*2^(n-i)
  end
  waveFun = zeros(Complex{Float32}, 2^n)
  waveFun[index+1] = 1
  H = Hamiltonian(1)
  print("Energy of these qubit = ", H[index+1,index+1], "\n")
  return waveFun
end

"""
H_init = [1 0 0 0;0 -1 0 0; 0 0 -1 0; 0 0 0 1]
H_fin = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
psi1 = [1;0;0;0]
diag_evolution(H_init, H_fin, psi1, 1, 1/19)
"""
end
