import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

corpus="""Hello Welcome,to Krish Naik's NLP Tutorials.
Please do watch the entire course! to become expert in NLP.
"""
docs = sent_tokenize(corpus)
for sentence in docs:
    print(sentence)

from nltk.tokenize import word_tokenize
word_tokenize(corpus)

for sentence in docs:
    print(word_tokenize(sentence))

from nltk.tokenize import wordpunct_tokenize
wordpunct_tokenize(corpus)

from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

tokenizer.tokenize(corpus)