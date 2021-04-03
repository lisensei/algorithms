import numpy as np
from numpy.random import default_rng
from numpy import e
'''Regularized Logistic Regression with gradient descent'''
'''Testin matrices are random'''

def logisticRegression(x,y,theta,alpha,lamd):
    m=x.shape[0]
    hp= 1 / (1 + e**(-(x@theta)))
    initialCost=-(y * (hp) - (np.ones((y.shape[0], 1)) - y) * np.log(np.ones((x.shape[0], 1)) - hp)).sum()
    i=0
    while True:
        temp=theta
        hofx = 1 / (1 + e**(-(x@theta)))
        theta=theta*(1-alpha*lamd/m)-alpha/m*(x.transpose()@(hofx-y))
        i+=1
        if abs((theta-temp).sum())<0.1**100:
            endCost = -(y * (hofx) - (np.ones((y.shape[0], 1)) - y) * np.log(np.ones((x.shape[0], 1)) - hofx)).sum()
            print("initial cost:",initialCost,"\nEnd cost:",endCost)
            print("Number of Iterations Executed:",i)
            break;
    return theta

def prediction(X):
    return 1 / (1 + e**(-(X@theta)))

rng=default_rng()
values=rng.standard_normal(1000)
x=rng.random((20,4))
y=np.random.randint(2,size=20).reshape((20,1))
theta=10*rng.random((4,1))
alpha=0.1
lamd=0
theta=logisticRegression(x,y,theta,alpha,lamd)
prediction=prediction(rng.random((1,4)))

print("X:",x,"\nY:",y.transpose(),"\nTheta:",theta)
print("prediction:",prediction)