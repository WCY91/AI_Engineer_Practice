import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

#change the like sex to the numeric
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])  #選擇哪一個column


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = my_data["Drug"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=3)

drug_tree = DecisionTreeClassifier(criterion="entropy",max_depth=5)
drug_tree.fit(X_train,y_train)

pred = drug_tree.predict(X_test)
print (pred[0:5])
print (y_test[0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, pred))
tree.plot_tree(drug_tree)
plt.show()








