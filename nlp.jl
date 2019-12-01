using CorpusLoaders
dataset = load(StanfordSentimentTreebank())

dataset[1, :]

using Flux

using Flux.Data

train = Data.Sentiment.train()

train[1]

s = "(3 (2 (2 Paul) (2 Bettany)) (3 (2 (2 is) (3 cool)) (2 .)))"
s = replace(s, r"[^ \n\(\)]+" => s -> "\"$s\"")
s = replace(s, " " => ", ")
ex = Meta.parse(s)
