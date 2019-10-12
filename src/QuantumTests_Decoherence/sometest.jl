function singleSpinOp(n)
  nStates = 2^n
  waveFunInterim = zeros(Complex{Float64}, nStates)
  waveFun = zeros(Complex{Float64}, nStates)
  fill!(waveFun,1)
  #display(waveFun)

  #global Hop = zeros(Complex{Float64}, nStates, nStates)
  for k = 0:n-1
    i1=2^k
    spinOp11 = 1;
    spinOp12 = 0;
    spinOp21 = 0;
    spinOp22 = 1;
    for l=0:2:nStates-1
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

  #for i::Int = 1:nStates
  #  global waveFun[i] = waveFunInterim[i]
  #end

  waveFun = waveFunInterim
  display(waveFun)

end




"""
    doubleSpinOp(delta, dt)
Double spin operator for time evolution. Based on eqn-70 from paper.
This funtion and singleSpinOp updates the wavefunction.
# Arguments
* `delta`: Time since initial configuration step.
* `dt`: duration of time steps.
"""
function doubleSpinOp(n)
  nStates = 2^n
  waveFunInterim = zeros(Complex{Float64}, nStates)
  waveFun = zeros(Complex{Float64}, nStates)
  fill!(waveFun,1)
  #display(waveFun)
    for k::Int = 0:n-1
      for l::Int = k+1:n-1
        nii::Int=2^k
        njj::Int=2^l


        dsOp11 = 1
        dsOp14 = 0
        dsOp22 = 1
        dsOp23 = 0
        dsOp32 = 0
        dsOp33 = 1
        dsOp41 = 0
        dsOp44 = 1

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
    display(waveFunInterim)
    #move the interim values of wavefunction to existing wavefunction.
    for i::Int = 1:nStates
      global  waveFun[i] = waveFunInterim[i]
    end
end
