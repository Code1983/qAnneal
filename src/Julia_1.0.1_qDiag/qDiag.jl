module qDiag

using LinearAlgebra

h_bar = 1.0545718001391127*10^-34
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
    #wavefun = diag_dt(H, wavefun, dt)
    display(time_step)
    display(wavefun)
    #display(H)
    wavefun = diag_dt(H, wavefun, dt)
    display("*******************************")
  end
  display("*******************************")
  #display(time_step)
  display(wavefun)
  return wavefun
end

end

"""
H_init = [1 0 0 0;0 -1 0 0; 0 0 -1 0; 0 0 0 1]
H_fin = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
psi1 = [1;0;0;0]
diag_evolution(H_init, H_fin, psi1, 1, 1/19)
"""
