#cd("QuantumTests")
using LinearAlgebra
using Distributed
using SharedArrays
@everywhere include("qAnnealv4.jl")
print("*******************************************************************************\n")
n=16
print("*******************************************************************************\n")
t=50
dt=0.01
a=qAnneal.anneal(n,t,dt)
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=45
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=40
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=35
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=30
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")


t=25
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=20
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=15
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")

t=5
dt=0.01
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.05
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.5
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=0.75
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=1
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
dt=2
b=qAnneal.anneal(n,t,dt)
print("**********, n=",n,",  t=",t,",  dt=",dt,", norm=",norm(norm.(a)-norm.(b)),"\n")
