using FileIO
using Flux
using Flux: onehotbatch
using Flux: @epochs, throttle, crossentropy, onecold
using Images
import Base.Iterators.partition
using Statistics: mean

batch_size = 10

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], Vector('A':'Z'))
    return (X_batch, Y_batch)
end


function readdata(folder)
    dataset = []
    for label in 'A':'Z'
        path = joinpath(folder, string(label))
        if isdir(path)
            files = readdir(path)
            for file in files
                push!(dataset, (joinpath(path, file), label))
            end
        end
    end

    pathset = [path for (path, _) in dataset]
    labelset = [label for (_, label) in dataset]

    img_arr = load.(pathset)
    arr = [Float64.(e) for e in img_arr]

    img_size = (40, 40)
    resize_img = imresize.(arr, img_size...)

    return resize_img, labelset
end

trainfolder = joinpath("/", "home", "xmeng", "projects", "julia", "train")
train_img, train_label = readdata(trainfolder)

set_len = length(train_label)

train = [make_minibatch(train_img, train_label, idx) for idx in  partition(1:set_len, batch_size)]

# Data should be stored in WHCN order (width, height, # channels, # batches). 
# https://fluxml.ai/Flux.jl/v0.10/models/layers/#Flux.Dense

# Bundle images together with labels and group into minibatchess

model = Chain(
    Conv((3, 3), 1 => 16, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, pad=(1, 1), relu),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(10*10*32, 26),
    softmax
)

# model(train[1][1])

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
loss(x, y) = crossentropy(model(x), y)

evalcb = () -> @show([accuracy(batch[1], batch[2]) for batch in train])

@epochs 20 Flux.train!(loss, params(model), train, ADAM(), cb = throttle(evalcb, 10))


testfolder = joinpath("/", "home", "xmeng", "projects", "julia", "test")
testset, test_label = readdata(testfolder)
testimg = cat(testset..., dims=4)
@show accuracy(testimg, onehotbatch(test_label, Vector('A':'Z')))

predict_test_label = onecold(model(testimg))

# predict_y = onecold(model(testX))
# actual_y = labelset
# using DataFrames
# df = DataFrame(actual=actual_y .-1 .+ 'A', predict=predict_y .- 1 .+ 'A')
# df[df.actual .!= df.predict , :]

# sklearn
using ScikitLearn

# This model requires scikit-learn. See
# http://scikitlearnjl.readthedocs.io/en/latest/models/#installation
@sk_import linear_model: LogisticRegression
@sk_import ensemble: RandomForestClassifier
@sk_import svm: LinearSVC
@sk_import svm: SVC

Xtest = hcat(reshape.(testset, :)...)'
ytest = string.(test_label)

Xtrain = hcat(reshape.(train_img, :)...)'
ytrain = string.(train_label)

skmodel = LinearSVC(C=0.1, max_iter=5000)
fit!(skmodel, Xtrain, ytrain)

skaccuracy = sum(predict(skmodel, Xtest) .== ytest) / length(ytest)
@show skaccuracy

sk_predict_label = predict(skmodel, Xtest)
using DataFrames
df = DataFrame(actual=ytest, predict=sk_predict_label)
df[df.actual .!= df.predict , :]