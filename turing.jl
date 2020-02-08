using CSV
using Plots
using DataFrames
using Turing
using StatsPlots
pyplot()

df = DataFrame(CSV.File("/home/xmeng/projects/CLSA/Problem2/DailyPrices.csv"))

using Flux

function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:6], 3, 2)
    b₁ = reshape(nn_params[7:9], 3)

    W₂ = reshape(nn_params[10:15], 2, 3)
    b₂ = reshape(nn_params[16:17], 2)

    Wₒ = reshape(nn_params[18:19], 1, 2)
    bₒ = reshape(nn_params[20:20], 1)

    return W₁, b₁, W₂, b₂, Wₒ, bₒ
end

function nn_forward(xs, nn_params::AbstractVector)
    W1, b1, W2, b2, Wo, bo = unpack(nn_params)
    nn = Chain(
        Dense(W1, b1, tanh),
        Dense(W2, b2, tanh),
        Dense(Wo, bo, relu)
    )
    return nn(xs)
end
alpha = 0.09
sig = sqrt(1.0 / alpha)

@model clsa(high, low, vol, avg_vol, highlow_ticks, beta, open, last, N) =  begin 
    # σₘ~ InverseGamma(1, 1)
    μ = (high + low) / 2
    nn_params ~ MvNormal(zeros(20), sig .* ones(20))

    for idx in (1:N-1)
        σₘ = nn_forward([beta[idx]; highlow_ticks[idx]], nn_params)[1]
        high[idx+1] ~ Normal(μ[idx], σₘ)
        low[idx+1] ~ Normal(μ[idx], σₘ)
    end

end

N = size(df)[1]
chain = sample(clsa(df.HIGH, df.LOW, df.VOLUME, df.AVGVOL30D, df.HIGHLOW_TICKS, df.EQY_BETA, df.OPEN, df.LAST, N), HMC(0.01, 10, :nn_params), 1000)
plot(chain)


a = chain[:nn_params].value
b = reshape(mean(a.data, dims=1), 20)
nn_forward(transpose([df.EQY_BETA df.HIGHLOW_TICKS]), b)

predict = Vector{Float64}(undef, 100)
mean_sigma = mean(chain[:σₘ].value)
predict[1] = rand(Normal(mu[end], mean_sigma))
for idx in 2:length(predict)
    predict[idx] = rand(Normal(predict[idx-1], mean_sigma)) 
end
