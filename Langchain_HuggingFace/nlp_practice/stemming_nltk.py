from nltk.stem import  PorterStemmer , RegexpStemmer,SnowballStemmer # often snowballstemmer is betten than the others

words=["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]
stemming = PorterStemmer()

for word in words:
    print(word+"---->"+stemming.stem(word))

print(stemming.stem('congratulations'))

reg_stemmer  = RegexpStemmer('ing$|s$|e$|able$',min = 4)
reg_stemmer.stem('eating')

reg_stemmer.stem('ingeating')

snowballsstemmer = SnowballStemmer('english')
for word in words:
    print(word+"---->"+snowballsstemmer.stem(word))

