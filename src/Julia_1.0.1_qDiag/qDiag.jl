module qDiag

using LinearAlgebra


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

  new_wavefun = v * exp(-im*dt*H_eig_diag) * v' * wavefun
  return new_wavefun/norm(new_wavefun)
end

"""
Calculate s from t.
currently, it is linear. Change this function for more complex evolution.
"""
function get_s(t)
  return t    #s is linear in t,
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
    wavefun = diag_dt(H, wavefun, dt)
    #display(time_step)
  end
  return wavefun
end

end
