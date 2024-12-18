import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("going",pos='v'))
words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]

for word in words:
    print(word+"---->"+lemmatizer.lemmatize(word,pos='v'))

print(lemmatizer.lemmatize("goes",pos='v'))
print(lemmatizer.lemmatize("fairly",pos='v'),lemmatizer.lemmatize("sportingly"))