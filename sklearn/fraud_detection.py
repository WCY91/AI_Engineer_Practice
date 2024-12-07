import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
raw_data = pd.read_csv(url)
print(raw_data)

#to make bigger dataset so inflate n times
n=10
big_data = pd.DataFrame(np.repeat(raw_data.values,n,axis=0),columns=raw_data.columns)
labels = big_data.Class.unique()
print("ending the data aug")
sizes = big_data.Class.value_counts().values
print(sizes)
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

big_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_data.iloc[:, 1:30])
data_matrix = big_data.values

X = data_matrix[:, 1:30]
y = data_matrix[:, 30]

X = normalize(X, norm="l1")
print('X.shape=', X.shape, 'y.shape=', y.shape)

X_train,X_test , y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y) #stratify=y 會這麼做是因為其為不平衡的兩個類別 所以希望在拆分時這個不平衡的比例也是要一致
# from sklearn.utils.class_weight import compute_sample_weight
# y = [1, 1, 1, 1, 0, 0]
# compute_sample_weight(class_weight="balanced", y=y)
# array([0.75, 0.75, 0.75, 0.75, 1.5 , 1.5 ])


w_train = compute_sample_weight('balanced',y_train)
sklearn_dt = DecisionTreeClassifier(max_depth=4,random_state=35)
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)
print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))


sklearn_svm = LinearSVC(class_weight='balanced',random_state=31,loss='hinge',fit_intercept=False)
t0 = time.time()
sklearn_svm.fit(X_train, y_train)
sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))
sklearn_pred = sklearn_svm.decision_function(X_test)
loss_sklearn = hinge_loss(y_test, sklearn_pred)
print("[Scikit-Learn] Hinge loss:   {0:.3f}".format(loss_sklearn))

