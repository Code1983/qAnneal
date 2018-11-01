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
times = np.linspace(0.0, 10.0, 20.0)

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