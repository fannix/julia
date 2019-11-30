using Statistics
using CuArrays
using Flux, Flux.Tracker, Flux.Optimise
using Metalhead, Images
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition

Metalhead.download(CIFAR10)
X = trainimgs(CIFAR10)
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000], 1:10)

image(x) = x.img
ground_truth(x) = x.ground_truth
image.(X[rand(1:end, 10)])

getarray(X) = float.(permutedims(channelview(X), (2, 3, 1)))
imgs = [getarray(X[i].img) for i in 1:50000]

train = gpu.([(cat(imgs[i]..., dims=4), labels[:, i]) for i in partition(1:49000, 1000)])
valset = 49001:50000
valX = cat(imgs[valset]..., dims=4) |> gpu
valY = labels[:, valset] |> gpu

m = Chain(
    Conv((5, 5), 3=>16, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 16 => 8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(200, 120),
    Dense(120, 84),
    Dense(84, 10),
    softmax) |> gpu


using Flux: crossentropy, Momentum

loss(x, y) = sum(crossentropy(m(x), y))
opt = Momentum(0.01)

accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))


using Flux: @epochs
epochs = 10

@epochs epochs Flux.train!(
    loss, params(m), train, opt,
    cb = () -> @show accuracy(valX, valY)
)