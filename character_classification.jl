
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

labelbatch = onehotbatch(labelset, string.(collect('A':'Z')))

using Images
img_size = (40, 40)
resize_img = imresize.(arr, img_size...)

img_batch = cat(resize_img..., dims=3)