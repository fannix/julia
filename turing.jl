using CSV
using Plots
using DataFrames
using Turing
using StatsPlots
pyplot()

df = DataFrame(CSV.File("/home/xmeng/projects/CLSA/Problem2/DailyPrices.csv"))

mu = (df.HIGH + df.LOW) / 2

@model clsa(μ, N) =  begin 
    σₘ~ InverseGamma(1, 1)
    for idx in (1:N-1)
        μ[idx+1] ~ Normal(μ[idx], σₘ)
    end
end

N = length(mu)
chain = sample(clsa(mu, N), HMC(0.01, 10), 1000)
plot(chain)

predict = Vector{Float64}(undef, 100)
mean_sigma = mean(chain[:σₘ].value)
predict[1] = rand(Normal(mu[end], mean_sigma))
for idx in 2:length(predict)
    predict[idx] = rand(Normal(predict[idx-1], mean_sigma)) 
end