import gensim
from gensim.models import Word2Vec,KeyedVectors
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')
vec_king = wv['king']

print(len(vec_king))
print(vec_king)
print(wv['cricket'])
print(wv.most_similar('cricket'))
print(wv.similarity("hockey","sports"))
vec=wv['king']-wv['man']+wv['woman']
print(vec)

print(wv.most_similar([vec]))