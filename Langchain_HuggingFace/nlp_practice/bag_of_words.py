import pandas as pd
messages=pd.read_csv('smsspamcollection/SMSSpamCollection',
                    sep='\t',names=["label","message"])

import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=100,binary=True)

X= cv.fit_transform(corpus).toarray()
import numpy as np

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
print(X)

print(cv.vocabulary_)