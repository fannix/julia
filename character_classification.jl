
using FileIO

folder = joinpath("/", "home", "xmeng", "projects", "julia")

trainset = []

for d in 'A':'Z'
    label = string(d)
    path = joinpath(folder, label)
    if isdir(path)
        files = readdir(path)
        for file in files
            push!(trainset, (joinpath(path, file), label))
        end
    end
end

pathset = [path for (path, _) in trainset]
labelset = [label for (_, label) in trainset]

img_arr = load.(pathset)
arr = [Float64.(e) for e in img_arr]

using Flux
using Flux: onehotbatch

using Images
img_size = (40, 40)
resize_img = imresize.(arr, img_size...)

# Data should be stored in WHCN order (width, height, # channels, # batches). 
# https://fluxml.ai/Flux.jl/v0.10/models/layers/#Flux.Dense

set_len = length(pathset)

import Base.Iterators.partition

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], string.(collect('A':'Z')))
    return (X_batch, Y_batch)
end

batch_size = 10

train = [make_minibatch(resize_img, labelset, idx) for idx in  partition(1:set_len, batch_size)]


model = Chain(
    Conv((3, 3), 1 => 16, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, pad=(1, 1), relu),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(10*10*32, 26),
    softmax
)

model(train[1][1])

loss(x, y) = crossentropy(model(x), y)


evalcb = () -> @show(loss(train[1][1], train[1][2]))

using Flux: @epochs

@epochs 10 Flux.train!(loss, params(model), train, ADAM(), cb = throttle(evalcb, 10))

# accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))