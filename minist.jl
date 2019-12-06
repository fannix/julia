using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using CuArrays

imgs = MNIST.images()
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()
Y = onehotbatch(labels, 0:9) |> gpu
m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

@show accuracy(X,Y)
