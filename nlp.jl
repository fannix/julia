# https://github.com/FluxML/Flux.jl/blob/master/src/data/sentiment.jl
# https://github.com/FluxML/model-zoo/blob/master/text/treebank/data.jl

using CorpusLoaders
dataset = load(StanfordSentimentTreebank())
dataset[1, :]


using Flux

using Flux.Data

traintree = Flux.Data.Sentiment.train()

# train[1]
# s = "(3 (2 (2 Paul) (2 Bettany)) (3 (2 (2 is) (3 cool)) (2 .)))"
# s = replace(s, r"[^ \n\(\)]+" => s -> "\"$s\"")
# s = replace(s, " " => ", ")
# ex = Meta.parse(s)

labels = map.(x -> x[1], traintree)
phrases = map.(x -> x[2], traintree)

using Flux: onehot
using Flux.Data: Tree, leaves, isleaf

tokens = vcat(map(leaves, phrases)...)
freq = Dict()

for e in tokens
    freq[e] = get(freq, e, 0) + 1
end

phrases = map.(t -> get(freq, t, 0) == 1 ? "UNK" : t, phrases)

alphabet = unique(tokens)
push!(alphabet, "UNK")

phrases_en = map.(t-> t == nothing ? t : onehot(t, alphabet), phrases)

labels_en = map.(t-> onehot(t, 0:4), labels)
train = map.(tuple, phrases_en, labels_en)

using Flux: crossentropy, ADAM, throttle, @show, @epochs

V = length(alphabet)
N = 50
embedding = randn(N, V)
Wₗ = Dense(N, N, tanh)
Wᵣ = Dense(N, N, tanh)
combine_child(a, b) = Wₗ(a) + Wᵣ(b)
sentiment = Chain(Dense(N, 5), softmax)

function forward(tree :: Tree)
    if (isleaf(tree))
        phrase, y = tree.value
        w = embedding * phrase
        ŷ = sentiment(w)
        loss = crossentropy(ŷ, y)
        return w, loss
    else
        _, y = tree.value
        w₁, loss₁ = forward(tree[1])
        w₂, loss₂ = forward(tree[2])
        w = combine_child(w₁, w₂)
        ŷ = sentiment(w)
        loss = crossentropy(ŷ, y)
        return w, loss₁ + loss + loss₂
    end
end

loss(tree) = forward(tree)[2]
opt = ADAM()
ps = params(embedding, Wₗ, Wᵣ, sentiment)
evalcb = () -> @show loss(train[1])

@epochs 10 Flux.train!(loss, ps, zip(train), opt, cb = throttle(evalcb, 10))