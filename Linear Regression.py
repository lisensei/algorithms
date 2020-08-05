import numpy as np
from numpy.linalg import inv
'''Regularized Linear Regression with gradient descent:'''
'''Testing matrices are random'''
def prediction(x,y,theta,alpha,lam,newdata):
    initalCost=((x@theta-y)**2).sum()
    m=x.shape[0]
    i=0
    '''update theta:'''
    while i<100000:
        theta=theta*(1-alpha*lam/m)-(alpha/m)*(x.transpose()@(x@theta-y))
        i+=1
    endCost = ((x@theta-y)**2).sum()

    print("Initial Cost:",initalCost,"\t End cost:",endCost)
    print("optimal theta values:\n",theta)
    return (x@theta).sum()

'''Testing matrices:'''
rand=np.random.default_rng(2)
x=rand.random((20,7))*10
x[:,0]=1
y=rand.random((20,1))*10
theta=rand.random((7,1))
alpha=0.00125
lam=10
newdata=rand.random((1,7))

'''The normal equation:'''
def theNormalEquation(x,y,lam):
    idm=np.identity(x.shape[1])
    neTheta=inv(((x.transpose()@x)+lam*idm))@x.transpose()@y
    return neTheta

print("Prediction computed using linear regression:",prediction(x,y,theta,alpha,lam,newdata))
print("Prediction computed using normal equation:",(x@theNormalEquation(x,y,lam)).sum())





