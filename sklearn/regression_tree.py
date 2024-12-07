import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")

print(data.isna().sum())
data.dropna(inplace=True)
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)
regression_tree = DecisionTreeRegressor(criterion = 'mse')
regression_tree.fit(X_train, Y_train)
regression_tree.score(X_test, Y_test)
prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)