# https://github.com/FluxML/Flux.jl/blob/master/src/data/sentiment.jl
# https://github.com/FluxML/model-zoo/blob/master/text/treebank/data.jl

using CorpusLoaders
dataset = load(StanfordSentimentTreebank())
dataset[1, :]


using Flux

using Flux.Data

traintree = Data.Sentiment.train()

# train[1]
# s = "(3 (2 (2 Paul) (2 Bettany)) (3 (2 (2 is) (3 cool)) (2 .)))"
# s = replace(s, r"[^ \n\(\)]+" => s -> "\"$s\"")
# s = replace(s, " " => ", ")
# ex = Meta.parse(s)

labels = map.(x -> x[1], traintree)
phrases = map.(x -> x[2], traintree)

using Flux.Data: Tree, leaves

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

using Flux: crossentropy, Momentum

