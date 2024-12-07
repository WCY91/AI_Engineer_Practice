import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df['custcat'].value_counts()
df.hist(column='income', bins=50)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat = neigh.predict(X_test)

print(accuracy_score(y_train,neigh.predict(X_train)))
print(accuracy_score(y_test,neigh.predict(X_test)))
