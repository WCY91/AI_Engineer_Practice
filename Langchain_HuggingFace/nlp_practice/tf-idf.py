import pandas as pd
messages=pd.read_csv('SpamClassifier-master/smsspamcollection/SMSSpamCollection',
                    sep='\t',names=["label","message"])

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordlemmatize=WordNetLemmatizer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordlemmatize.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
x = tfidf.fit_transform(corpus).toarray()
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

# n-grams
tfidf = TfidfVectorizer(max_features=1000,ngram_range=(2,2))
X=tfidf.fit_transform(corpus).toarray()
print(tfidf.vocabulary_)

'''
Lemmatisation is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, 
and therefore cannot discriminate between words which have different meanings depending on part of speech. 
However, stemmers are typically easier to implement and run faster,
 and the reduced accuracy may not matter for some applications.
'''