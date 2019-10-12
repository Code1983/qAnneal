module qAnneal

#import Gadfly for plots
#using Gadfly

using DelimitedFiles
using LinearAlgebra
using Distributed
using SharedArrays
using Random

print("Added shared arrays. Number of procs = ",nprocs(),"\n")

#create the initial h and J matrix based on number of qubits
const jzStart = readdlm("./Jz_init.csv",',',Float32)
const jxStart = readdlm("./Jx_init.csv",',',Float32)
const jyStart = readdlm("./Jy_init.csv",',',Float32)
const hxStart = readdlm("./hx_init.csv",',',Float32)
const hyStart = readdlm("./hy_init.csv",',',Float32)
const hzStart = readdlm("./hz_init.csv",',',Float32)
#read the J and h files and create corresponding matrices
#This corresponds to the final influences to the system.const jz = readdlm("./Jz.txt")
const jz = readdlm("./Jz.csv",',',Float32)
const jx = readdlm("./Jx.csv",',',Float32)
const jy = readdlm("./Jy.csv",',',Float32)
const hz = readdlm("./hz.csv",',',Float32)
const hx = readdlm("./hx.csv",',',Float32)
const hy = readdlm("./hy.csv",',',Float32)

"""
Normalizes the wave function.
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
Initializes the spin system.
"""
function init(nStates::Int, checkpoint::Int, time_start::Float64, waveFun=missing)


  if waveFun === missing
    waveFun = SharedArray{Complex{Float32},1}(nStates)
    fill!(waveFun,1)
  end

  print("checkpoint Details (Input_chkpoint_step,time_start): ",checkpoint, " , ", time_start, "\n")
  if checkpoint != 0 && time_start != 0.0
    print("reading checkpointed wave files  \n")
    try
      waveFun = readdlm("real_part.txt",Float32)+im*readdlm("imag_part.txt",Float32)
    catch
      print("Warning reading checkpointed wave files  \n")
    end
  end

  waveFun = normWaveFun(waveFun)
  println("Initialization done...")
  return waveFun
end

"""
Single spin operator for time evolution. Based on eqn-68 from the paper.
This funtion and doubleSpinOp updates the wavefunction.
"""
function singleSpinOp(n, nStates, delta, dt, waveFun, waveFunInterim)
  #waveFunInterim = zeros(Complex{Float32}, nStates)
  for k = 0:n-1
    i1=2^k
    hxi::Float32 = (1-delta)*hxStart[k+1] + delta*hx[k+1]   #The +1 is because indexing in julia is 1-based
    hyi::Float32 = (1-delta)*hyStart[k+1] + delta*hy[k+1]
    hzi::Float32 = (1-delta)*hzStart[k+1] + delta*hz[k+1]
    #based on equation 68
    hi::Float32 = sqrt(hxi*hxi + hyi*hyi + hzi*hzi)
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
    @sync @distributed for l=0:2:nStates-1
        i2= l & i1;
        i::Int = l - i2 + i2/i1 + 1;  #The +1 is because indexing in julia is 1-based
        j::Int = i + i1;
        #print("====",k," ",i," ",j," ",spinOp11," ",spinOp22, "\n")

        waveFunInterim[i] += spinOp11*waveFun[i] + spinOp12*waveFun[j]
        waveFunInterim[j] += spinOp21*waveFun[i] + spinOp22*waveFun[j]
    end
  end
  return normWaveFun(waveFunInterim)
end




"""
Double spin operator for time evolution. Based on eqn-70 from paper.
"""
function doubleSpinOp(n, nStates, delta, dt, waveFun, waveFunInterim)
  #waveFunInterim = zeros(Complex{Float32}, nStates)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1
        nii::Int=2^k
        njj::Int=2^l
        jxij=(1-delta)*jxStart[k+1,l+1] + delta*jx[k+1,l+1]
        jyij=(1-delta)*jyStart[k+1,l+1] + delta*jy[k+1,l+1]
        jzij=(1-delta)*jzStart[k+1,l+1] + delta*jz[k+1,l+1]

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

        @sync @distributed for m::Int = 0:4:nStates-1
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
    return normWaveFun(waveFunInterim)
end




"""
calculates the global energy of the system
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
        jxij=(1-delta)*jxStart[k+1,l+1] + delta*jx[k+1,l+1]
        jyij=(1-delta)*jyStart[k+1,l+1] + delta*jy[k+1,l+1]
        jzij=(1-delta)*jzStart[k+1,l+1] + delta*jz[k+1,l+1]
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

function energyByMatrix(delta::Float64=1; H=missing, waveFun=missing)
  if H === missing
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


"""
Calculates the Hamiltonian of the system.
"""
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
        jxij=(1-delta)*jxStart[k+1,l+1] + delta*jx[k+1,l+1]
        jyij=(1-delta)*jyStart[k+1,l+1] + delta*jy[k+1,l+1]
        jzij=(1-delta)*jzStart[k+1,l+1] + delta*jz[k+1,l+1]
        for m::Int = 0:4:nStates-1
            n3::Int = m & njj;
            n2::Int = m-n3+(n3+n3)/njj;
            n1::Int = n2 & nii;
            n0::Int = n2 - n1+n1/nii +1; # The +1 is because indexing in julia is 1-based
            n1=n0+nii;
            n2=n0+njj;
            n3=n1+njj;

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
Currently, it is linear. Change this function for more complex evolution.
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
function anneal(nQ::Int, t=1.0, dt=0.1, checkpoint=0, initWaveFun=missing)
    println("Annealing started...")

    #initialize the system
    nStates::Int = 2^nQ     #total number of states

    checkpoint_count::Int = 0
    time_start = 0.0
    if checkpoint != 0
      print("Reading checkpoint  \n")
      try
        time_start = readdlm("checkpoint.txt")[1] + dt
      catch
        print("Warning reading checkpoint  \n")
      end
    end

    waveFun = init(nStates, checkpoint, time_start, initWaveFun)
    waveFunInterim = SharedArray{Complex{Float32},1}(nStates)
    σ = []
    T = []

    for time_step = time_start:dt:t-dt
        #delta = i/nSteps
        #display(time_step)
        #display(waveFun)
        s = get_s(time_step/t)
        #print(s," ",dt, "\n ")
        @sync @distributed for i::Int = 1:nStates
         waveFunInterim[i] = 0.0+0.0im
        end
        waveFun = singleSpinOp(nQ, nStates, s, dt, waveFun, waveFunInterim)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        #singleSpinOp(s, dt)
        @sync @distributed for i::Int = 1:nStates
         waveFunInterim[i] = 0.0+0.0im
        end
        waveFun = doubleSpinOp(nQ, nStates, s, dt, waveFun, waveFunInterim)
        #doubleSpinOp(s, dt)
        #doubleSpinOp(s, dt)
        @sync @distributed for i::Int = 1:nStates
         waveFunInterim[i] = 0.0+0.0im
        end
        waveFun = singleSpinOp(nQ, nStates, s, dt, waveFun, waveFunInterim)
        #display("*******************************")
        #energy[i+1]=energySys(delta)
        checkpoint_count = checkpoint_count + 1

        #print(time_step)
        sig = qAnneal.decoherence(waveFun, [4,8])
        push!(σ,sig)
        push!(T,time_step)
        #print(time_step,"\t",sig,"\n")

        if checkpoint_count >= checkpoint && checkpoint != 0
          open("checkpoint.txt", "w") do chkpnt_file
            print("Checkpointed \n")
            writedlm(chkpnt_file, time_step)
          end;
          open("real_part.txt", "w") do real_part
            writedlm(real_part, real(waveFun))
          end
          open("imag_part.txt", "w") do imag_part
            writedlm(imag_part, imag(waveFun))
          end
          break
        end
    end

    println("Annealing done...")
    #return waveFun
    return T,σ

    #plot(x=0:nSteps,y=real(energy),Geom.line)

end






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
  waveFun = init(nStates, initWaveFun)
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
print high probabbility a given state of qubit.
qState = state Vector
n = total number of qubitsToWavefun
top = limits the number of high probability states.
"""
function qDisplay(qState, n, top)
    qindex = sortperm(abs2.(qState),rev=true)

    print("Math way of representing with probability amplitude. \n")
    for i=1:top
        print( "(", qState[qindex[i]], ") |",lpad(decToBin(qindex[i]-1),n,'0'),"> \n")
    end
    print("\n\n")
    print("Physics way of representing with probabilities. \n")
    for i=1:top
        print( "(", abs2(qState[qindex[i]]), ") |",replace(replace(reverse(lpad(decToBin(qindex[i]-1),n,'0')),'0' => '↑'),'1' => '↓'),"> \n")
    end
end

"""
Sort by energy
qState = state Vector
n = total number of qubitsToWavefun
top = limits the number of high probability states.
"""
function sortEnergy(qE, n, nTopBottom)
    qindex = sortperm(real(qE),rev=true)
    sort!(real(qE),rev=true)
    print("Energy \n Top \n")
    for i=1:nTopBottom
        print( "(", qE[qindex[i]], ") |",replace(replace(reverse(lpad(decToBin(qindex[i]-1),n,'0')),'0' => '↑'),'1' => '↓'),"> \n")
    end
    #print(".\n.\n.\n")
    print("Bottom\n")
    for i=2^n-nTopBottom+1:2^n
        print( "(", qE[qindex[i]], ") |",replace(replace(reverse(lpad(decToBin(qindex[i]-1),n,'0')),'0' => '↑'),'1' => '↓'),"> \n")
    end
end

"""
Calculates the Diagonal elements of the Hamiltonian of the system.
This can be used to find the energy of a pure state.
E = DiagOfHamiltonian(22)

"""
function DiagOfHamiltonian(n)

    nStates = 2^n
    delta = 1

    # variable to hold H
    H_diag = zeros(Complex{Float32}, nStates)
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

        H_diag[i] += hzi
        H_diag[j] += - hzi
      end
    end

    # Calculate (Σ Jij σi σj)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1
        nii::Int=2^k
        njj::Int=2^l
        jxij=(1-delta)*jxStart[k+1,l+1] + delta*jx[k+1,l+1]
        jyij=(1-delta)*jyStart[k+1,l+1] + delta*jy[k+1,l+1]
        jzij=(1-delta)*jzStart[k+1,l+1] + delta*jz[k+1,l+1]
        for m::Int = 0:4:nStates-1
            n3::Int = m & njj;
            n2::Int = m-n3+(n3+n3)/njj;
            n1::Int = n2 & nii;
            n0::Int = n2 - n1+n1/nii +1; # The +1 is because indexing in julia is 1-based
            n1=n0+nii;
            n2=n0+njj;
            n3=n1+njj;

            H_diag[n0] += jzij
            H_diag[n1] += -jzij
            H_diag[n2] += -jzij
            H_diag[n3] += jzij

        end
      end
    end


    return -H_diag

end


"""
calculate decoherence.
The code handles system that has consecutive system and environment qubits.
"""
function decoherence(state, dims)
    sys_qubits = dims[1]
    env_qubits = dims[2]

    sys_dim = 2^sys_qubits
    env_dim = 2^env_qubits

    σ_sqr=0

    for i = 1:sys_dim-1
      for j = i+1:sys_dim
          for p = 1:env_dim
            σ_sqr=  σ_sqr + abs2(conj(state[i*sys_dim+p])*state[j*sys_dim+p])
          end
      end
    end

    σ = sqrt(σ_sqr)

end

"""
Calculate thermalization
"""
function theralization(state, E, dims)
    sys_start = dims[1]
    sys_qubits = dims[2]
    env_start = dims[3]
    env_qubits = dims[4]

    sys_dim = 2^sys_qubits
    env_dim = 2^env_qubits

    for i = 1:sys_dim
        for p = 1:env_dim
          ρ_ii[i] =  ρ_ii[i] + conj(state[i*sys_dim+p])*state[i*sys_dim+p]
        end
    end

    b_num=0
    b_den=0
    for i=1:sys_dim
      for j = i+1:sys_dim
        if E[i] != E[j]
          b_num =  b_num + ln(ρ_ii[i]) - ln(ρ_ii[j])/(E[j]-E[i])
          b_den = b_den + 1
        end
      end
  end
  b - b_num/b_den
end


"""
Calculate Random Initial StackTraces
"""
function randomState(n ::Int)
    nState = 2^n
    r0 = rand(nState)
    r1 = rand(nState)
    d = SharedArray{Complex{Float32}}(sqrt.(-2*log.(r0)).*cos.(2*pi*r1) + im*sqrt.(-2*log.(r0)).*sin.(2*pi*r1))
    #return normWaveFun(d)
end


"""
Calculates canonical thermal state.
Inputs - H: Hamiltonian, wavefun: wavefunction |Ψ(0)>, T:tempreture
|(Σ Jij σi σj)|Ψ(β)> = (e^-βH/2)|Ψ(0)> / <Ψ(0)|(e^-βH/2)|Ψ(0)>^0.5
returns canonical thermal state |Ψ(β)>
"""
function cannonical_state(H, wavefun, T)
  if (!ishermitian(H))
    display("H is not hermitian.")
    return 0
  end

  H_herm = Hermitian(H)

  eig = eigen(H_herm)
  H_eig_diag = Diagonal(eig.values)
  v = eig.vectors

  new_wavefun = v * exp(-H_eig_diag/(2*T)) * v' * wavefun
  return new_wavefun/norm(new_wavefun)
end


"""
H_init = [1 0 0 0;0 -1 0 0; 0 0 -1 0; 0 0 0 1]
H_fin = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
psi1 = [1;0;0;0]
diag_evolution(H_init, H_fin, psi1, 1, 1/19)
"""

"""
Sample code on using this module.
@everywhere include("qAnnealv4.jl")
@time b=qAnneal.anneal(2,1,0.01)
qAnneal.diag_evolve(2,1,0.01)

using LinearAlgebra
using Distributed
using SharedArrays
addprocs(1)
@everywhere include("qAnnealv5.jl")
"""

end
