#Author: Li Sensei
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets

plt.ion()
xn,yn=datasets.make_regression(100,1,noise=20)
plt.plot(xn,yn,'ro')

x=torch.tensor(xn.astype(np.float32))
y=torch.tensor(yn.astype(np.float32))
y=y.view(y.shape[0],1)

inputSize=x.shape[1]
outputSize=1

model=nn.Linear(in_features=inputSize,out_features=outputSize,bias=True)
loss=nn.MSELoss()
lr=0.01
optimizer=torch.optim.SGD(model.parameters(),lr)
ln,=plt.plot(xn, yn, 'ro')
for i in range(200):
    yPred=model(x)
    l=loss(yPred,y)
    l.backward()

    q,=plt.plot(xn,yPred.detach().numpy(),'b')
    plt.pause(0.05)
    q.remove()

    optimizer.step()
    optimizer.zero_grad()


plt.close()





