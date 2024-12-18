# -*- coding: utf-8 -*-
"""Deep Learning PyTorch.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CYxbg-lGWZk_MDRsQ4sWvsENrwy0DXZe

# Multiple Linear Regression
"""

from torch import nn
import torch
torch.manual_seed(1)

w = torch.tensor([[2.0],[3.0]],requires_grad=True)
b = torch.tensor([[1.0]],requires_grad=True)

def forward(x):
  return torch.mm(w,x)+b

x = torch.tensor([[1.0,2.0]])
yhat = forward(x)
print("The result: ", yhat)

X = torch.tensor([[[1.0,1.0],[1.0,2.0],[1.0,3.0]]])
# yhat = forward(X)

model = nn.Linear(in_features=2,out_features=1)
yhat = model(x)

# build custom model
class linear_regression(nn.Module):

  def __init__(self,input_size,output_size):
    super(linear_regression,self).__init__()
    self.linear = nn.Linear(input_size,output_size)

  def forward(self,x):
    yhat = self.linear(x)
    return yhat


model = linear_regression(2,1)
print("The parameters: ", list(model.parameters()))
print("The parameters: ", model.state_dict())
yhat = model(x)
print("The result: ", yhat)

# # 將data傳入model進行forward propagation
# # 計算loss
# # 清空前一次的gradient
# # 根據loss進行back propagation，計算gradient
# # 做gradient descent
# output = model(batch_x)
# loss = criterion(output, batch_y)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# Pytorch不幫你自動清空gradient，而是要你呼叫optimizer.zero_grad()來做這件事是因為，這樣子你可以有更大的彈性去做一些黑魔法，畢竟，誰規定每一次iteration都要清空gradient?

# 試想你今天GPU的資源就那麼小，可是你一定要訓練一個很大的model，然後如果batch size不大又train不起來，那這時候該怎麼辦?

# 雖然沒有課金解決不了的事情，如果有，那就多課一點…不是，這邊提供另外一種設計思維:

# 你可以將你的model每次都用小的batch size去做forward/backward，但是在update的時候是多做幾個iteration在做一次。

# 這個想法就是梯度累加(gradient accumulation)，也就是說我們透過多次的iteration累積backward的loss，然後只對應做了一次update，間接的做到了大batch size時候的效果。

# for idx, (batch_x, batch_y) in enumerate(data_loader):
#     output = model(batch_x)
#     loss = criterion(output, batch_y)

#     loss = loss / accumulation_step
#     loss.backward()

#     if (idx % accumulation_step) == 0:
#         optimizer.step() # update
#         optimizer.zero_grad() # reset

from torch import nn,optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset,DataLoader

def Plot_2D_Plane(model,dataset,n=0):
  print(model.state_dict())
  w1 = model.state_dict()['linear.weight'].numpy()[0][0]
  w2 = model.state_dict()['linear.weight'].numpy()[0][1]
  b = model.state_dict()['linear.bias'].numpy()

  x1 = dataset.x[:,0].view(-1,1).numpy() #-1代表自動判斷維度
  x2 = dataset.x[:,1].view(-1,1).numpy()
  y = dataset.y.numpy()

  X,Y = np.meshgrid(np.arange(x1.min(),x1.max(),0.05),np.arange(x2.min(),x2.max(),0.05))
  yhat = w1*X + w2*Y +b

  fig = plt.figure()
  ax =  fig.add_subplot(projection='3d')
  ax.plot(x1[:,0],x2[:,0],y[:,0],'ro',label='y')
  ax.plot_surface(X,Y,yhat)

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('y')
  plt.title('estimated plane iteration:' + str(n))
  ax.legend()

  plt.show()

class Data2D(Dataset):
  def __init__(self):
    self.x = torch.zeros(20,2)
    self.x[:,0] = torch.arange(-1,1,0.1)
    self.x[:,1] = torch.arange(-1,1,0.1)
    self.w = torch.tensor([[1.0],[1.0]])
    self.b  =1
    self.f = torch.mm(self.x,self.w) + self.b
    self.y = self.f + 0.1* torch.randn((self.x.shape[0],1))
    self.len = self.x.shape[0]

  def __getitem__(self,index):
    return self.x[index],self.y[index]

  def __len__(self):
    return self.len

data_set = Data2D()

class linear_regression(nn.Module):
  def __init__(self,input_size,output_size):
    super(linear_regression,self).__init__()
    self.linear = nn.Linear(in_features = input_size,out_features=output_size)

  def forward(self,x):
    yhat = self.linear(x)
    return yhat

model = linear_regression(2,1)
optimizer = optim.SGD(model.parameters(),lr=0.1)
criterion = nn.MSELoss()
train_loader = DataLoader(dataset=data_set, batch_size=2)

LOSS = []
print("Before Training: ")
Plot_2D_Plane(model, data_set)
epochs = 100

def train_model(epochs):
  for epoch in range(epochs):
    for x,y in train_loader:
      yhat = model(x)
      loss = criterion(yhat,y)
      LOSS.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

train_model(epochs)
print("After Training: ")
Plot_2D_Plane(model, data_set, epochs)
with torch.no_grad():
  plt.plot(LOSS)
  plt.xlabel("Iterations ")
  plt.ylabel("Cost/total loss ")
  plt.show()

from torch import nn
import torch
torch.manual_seed(1)

class linear_regression(nn.Module):
  def __init__(self,input_size,output_size):
    super(linear_regression,self).__init__()
    self.linear = nn.Linear(input_size,output_size)

  def forward(self,x):
    return self.linear(x)


model = linear_regression(1,10)
model(torch.tensor([1.0]))
print(list(model.parameters()))

x=torch.tensor([[1.0]])
print(model(x))

X=torch.tensor([[1.0],[1.0],[3.0]])
print(model(X))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim,nn
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

torch.manual_seed(1)
class Data(Dataset):
  def __init__(self):
    self.x = torch.zeros(20,2)
    self.x[:,0] = torch.arange(-1,1,0.1)
    self.x[:,1] = torch.arange(-1,1,0.1)
    self.w = torch.tensor([ [1.0,-1.0],[1.0,3.0]])
    self.b = torch.tensor([[1.0,-1.0]])
    self.f = torch.mm(self.x,self.w)+self.b
    self.y = self.f+0.001*torch.randn((self.x.shape[0],1))
    self.len=self.x.shape[0]

  def __getitem__(self,index):
    return self.x[index] , self.y[index]

  def __len__(self):
    return self.len

data_set=Data()
class linear_regression(nn.Module):
  def __init__(self,input_size,output_size):
    super(linear_regression,self).__init__()
    self.linear = nn.Linear(input_size,output_size)

  def forward(self,x):
    return self.linear(x)

model=linear_regression(2,2)
optimizer = optim.SGD(model.parameters(),lr = 0.1)
criterion = nn.MSELoss()
train_loader=DataLoader(dataset=data_set,batch_size=5)

LOSS = []
epochs = 100
for epoch in range(epochs):
  for x,y in train_loader:
    yhat = model(x)
    loss = criterion(yhat,y)
    LOSS.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
  plt.plot(LOSS)
  plt.xlabel("iterations ")
  plt.ylabel("Cost/total loss ")
  plt.show()



"""# Logistic Regression"""

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(2)
z = torch.arange(-100,100,0.1).view(-1,1) #view的作用是reshape
sig = nn.Sigmoid()
yhat = sig(z)

plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)

model = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())
yhat = model(x)
print("The prediction: ", yhat)
yhat = model(X)
print("The prediction: ", yhat)

x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
print('x = ', x)
print('X = ', X)
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())
yhat = model(x)
print("The prediction: ", yhat)
yhat = model(X)
print("The prediction: ", yhat)

class logistic_regression(nn.Module):
  def __init__(self,n_inputs):
    super(logistic_regression,self).__init__()
    self.linear = nn.Linear(n_inputs,1)

  def forward(self,x):
    return self.linear(x)

x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
print('x = ', x)
print('X = ', X)
model = logistic_regression(2)
print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())
yhat = model(x)
print("The prediction result: \n", yhat)
yhat = model(X)
print("The prediction result: \n", yhat)

x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
print('x = ', x)
print('X = ', X)
yhat = model(x)
print("The prediction result: \n", yhat)
yhat = model(X)
print("The prediction result: \n", yhat)

