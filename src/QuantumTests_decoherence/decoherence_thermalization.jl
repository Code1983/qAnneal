using Random

"""
calculate decoherence.
The code handles system that has consecutive system and environment qubits.
"""
function decoherence(state, dims)
    sys_start = dims[1]
    sys_qubits = dims[2]
    env_start = dims[3]
    env_qubits = dims[4]

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
    r0 = rand(n)
    r1 = rand(n)
    d = sqrt.(-2*log.(r0)).*cos.(2*pi*r1) + im*sqrt.(-2*log.(r0)).*sin.(2*pi*r1)
    normWaveFun(d)
end
