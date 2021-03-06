#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 07:19:27 2018

@author: malay
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

H = 2 * np.pi * 0.1 * qt.sigmax()
psi0 = qt.basis(2, 0)
times = np.linspace(0.0, 1.0, 20.0)

#creating a callback for Hamiltonian
def H_call(s, args):
  H_init = qt.sigmaz()
  H_fin  = qt.sigmax()
  H_val  = (1-s)*H_init + s*H_fin
  #print(H_val)
  #print(s)
  #print("*********")
  return H_val

'''
#calculates expectation values of sigmaz
result = qt.mesolve(H, psi0, times, [], [qt.sigmaz()])
#result = qt.mesolve(H, psi0, times, [], [qt.sigmaz(), qt.sigmay()])
print(result)
print(result.expect)

#calculates state evolution
result1 = qt.mesolve(H, psi0, times, [])
print(result1)
print(result1.states[0])

#calculates expectation values of sigmaz
result2 = qt.sesolve(H, psi0, times, [], [qt.sigmaz()])
#result = qt.mesolve(H, psi0, times, [], [qt.sigmaz(), qt.sigmay()])
print(result2)
print(result2.expect)

#calculates state evolution
result3 = qt.sesolve(H, psi0, times, [])
print(result3)
print(result3.states[0])
'''

#calculates state evolution
result4 = qt.sesolve(H_call, psi0, times, [])
#print(result4)
#print(result4.states[0])

#multi cubit operation
psi1 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
times = np.linspace(0.0, 1.0, 20.0)

#creating a callback for Hamiltonian
def H_two_call(s, args):
  H_init = qt.tensor([qt.sigmaz(), qt.sigmaz()])
  H_fin  = qt.tensor([qt.sigmax(), qt.sigmax()])
  H_val  = (1-s)*H_init + s*H_fin
  #print(H_val)
  #print(s)
  #print("*********")
  return H_val

result5 = qt.sesolve(H_two_call, psi1, times, [])

# It seems mesolve is calling calling sesolve on the backend
# This is confimed based on output from printing the results


'''
H = 2 * np.pi * 0.1 * qt.sigmax()
psi0 = qt.basis(2, 0)
times = np.linspace(0.0, 10.0, 100)
result = qt.mesolve(H, psi0, times, [], [qt.sigmaz(), qt.sigmay()])
fig, ax = plt.subplots()
ax.plot(result.times, result.expect[0]);
ax.plot(result.times, result.expect[1]);
ax.set_xlabel('Time');
ax.set_ylabel('Expectation values');
ax.legend(("Sigma-Z", "Sigma-Y"));
plt.show()


times = [0.0, 1.0]
result = qt.mesolve(H, psi0, times, [], [])
result.states
'''
