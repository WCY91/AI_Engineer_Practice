# -*- coding: utf-8 -*-
"""part_all.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vS2yFNW2iVfc4YPjc_mMFWxu482UyPZx
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

concrete_data_columns = concrete_data.columns

concrete_data = concrete_data.dropna() # data clean
target_label = concrete_data['Strength']
predict_data = concrete_data.drop(columns = 'Strength')

import keras

from keras.models import Sequential
from keras.layers import Dense

#A-1
X_train, X_test, y_train, y_test = train_test_split(predict_data, target_label, test_size=0.3, random_state=42)
def regression_model():
  model= Sequential()
  model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
  model.add(Dense(1))

  # compile model
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model

model = regression_model()

#A-2
model.fit(X_train, y_train, validation_split=0.1, epochs=50, verbose=2) # verbose is to show the progress of model train

y_pred = model.predict(X_test).flatten()
# 計算測試集上的均方誤差
mse = mean_squared_error(y_test, y_pred)
print(mse)

mse_list = []
def repeat_n_times_train(n = 50):
  for i in range(n):
    print(f"in {i+1} times train")
    model = regression_model()
    model.fit(X_train, y_train, validation_split=0.1, epochs=50, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test,y_pred)
    print(f"in {i+1} times train mse is {mse}")
    mse_list.append(mse)
repeat_n_times_train(50)
print(mse_list)

mse_mean = np.mean(mse_list)
mse_std = np.std(mse_list)
print(mse_list)
print(f"50 epoch the mean mse is {mse_mean} the std mse is {mse_std}")

# part b
data_preprocess = (predict_data - np.mean(predict_data)) / np.std(predict_data)
X_train, X_test, y_train, y_test = train_test_split(data_preprocess, target_label, test_size=0.3, random_state=42)

mse_list = []
repeat_n_times_train(50)
print(mse_list)

mse_mean = np.mean(mse_list)
mse_std = np.std(mse_list)
print(mse_list)
print(f"in part B after data cleaning 50 epoch the mean mse is {mse_mean} the std mse is {mse_std}")

#part c
result_c = []
def repeat_n_times_train_with_epoch_100(n = 50):
  for i in range(n):
    print(f"in {i+1} times train")
    model = regression_model()
    model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test,y_pred)
    print(f"in {i+1} times train mse is {mse}")
    result_c.append(mse)
repeat_n_times_train_with_epoch_100(50)
print(result_c)
mse_mean = np.mean(result_c)
mse_std = np.std(result_c)
print(f"in part C after data cleaning 50 epoch 100 the mean mse is {mse_mean} the std mse is {mse_std}")

#part d
def regression_model_with_3_hidden_layer():
  model= Sequential()
  model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))

  # compile model
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model
result_d = []
def repeat_n_times_train_with_part_d(n = 50):
  for i in range(n):
    print(f"in {i+1} times train")
    model = regression_model_with_3_hidden_layer()
    model.fit(X_train, y_train, validation_split=0.1, epochs=50, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test,y_pred)
    print(f"in {i+1} times train mse is {mse}")
    result_d.append(mse)
repeat_n_times_train_with_part_d(50)
print(result_d)
mse_mean = np.mean(result_d)
mse_std = np.std(result_d)
print(f"in part D  the mean mse is {mse_mean} the std mse is {mse_std}")

