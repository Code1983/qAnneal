@everywhere include("qAnnealv4.jl")
@time b=qAnneal.anneal(2,1,0.01)
display(b)
