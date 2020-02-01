using CSV
using Plots
using DataFrames
using Turing

df = DataFrame(CSV.File("/home/xmeng/projects/CLSA/Problem2/DailyPrices.csv"))

@model clsa(μ, N) =  begin 
    σₘ~ InverseGamma(1, 1)
    for idx in (1:N-1)
        μ[idx+1] ~ Normal(μ[idx], σₘ)
    end
end

