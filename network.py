import numpy as np
from numpy import e
import struct
import random
from charclass import charRec
import mnist_loader
def logi(z):
    return 1.0/(1.0+np.exp(-z))

def dlogi_dz(z):
    return logi(z)*(1.0-logi(z))

class network:
    def __init__(self,size):
          self.l=len(size)
          self.weights=[np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
          self.biases=[np.random.randn(x,1) for x in size[1:]]



    def forwardprop(self,a):
        for w,b in zip(self.weights,self.biases):
            a=logi(w@a+b)
        return a
    def backprop(self,x,y):
        dWeights=[np.zeros(w.shape) for w in self.weights]
        dBiases=[np.zeros(b.shape) for b in self.biases]
        a=x
        deltas=[]
        activisions=[x]
        zs=[]
        for w,b in zip(self.weights,self.biases):
            z=w@a+b
            zs.append(z)
            a=logi(z)
            activisions.append(a)
        deltaL=(activisions[-1]-y)*dlogi_dz(zs[-1])
        deltas.append(deltaL)
        delta=deltaL

        for i in range(1,self.l-1):
            delta=(self.weights[-i].transpose()@delta)*dlogi_dz(zs[-i-1])
            deltas.append(delta)
        deltas.reverse()
        dWeights=[d@(a.transpose()) for d, a in zip(deltas,activisions[:-1])]
        dBiases=[d for d in deltas]
        return  (dWeights,dBiases)
    def gradientDescent(self,x,y,lamda,numOfIterations):

        m=len(x)
        ite=0
        while ite<numOfIterations:
            print("Iterations",ite,": "+str(self.testing(testData)))
            dwadder=[np.zeros(w.shape) for w in self.weights]
            dbadder=[np.zeros(b.shape) for b in self.biases]
            for i,j in zip(x,y):
                dWeights,dBaise=self.backprop(i,j)
                dwadder=[w1+w2 for w1,w2 in zip(dwadder,dWeights)]
                dbadder=[b1+b2 for b1,b2 in zip(dbadder,dBaise)]
            self.weights=[w-lamda/m*dw for w,dw in zip(self.weights,dwadder)]
            self.biases=[b-lamda/m*db for b,db in zip(self.biases,dbadder)]
            ite+=1
        return (self.weights,self.biases)

    def stochasticGradientDescent(self,trainingData,lamda,batchSize,numberOfIterations):
        m=len(trainingData)
        iteCount=0
        while iteCount<numberOfIterations:
            result=self.testing(testData)
            print("Iteration:",iteCount,"Precision:",result)
            random.shuffle(trainingData)
            c=0
            k=batchSize
            if result>0.95:
                k=m

            while c<m:
                dwa=[np.zeros(w.shape) for w in self.weights]
                dba=[np.zeros(b.shape) for b in self.biases]
                for i,j in trainingData[c:c+k]:
                    dWeights,dBiases=self.backprop(i,j)
                    dwa=[x + y for x,y in zip(dwa,dWeights)]
                    dba = [x + y for x, y in zip(dba, dBiases)]
                self.weights=[w - lamda/k*(dw)  for w, dw in zip(self.weights, dwa)]
                self.biases = [b -lamda/k*(db)  for b, db in zip(self.biases, dba)]
                c=c+k
            iteCount+=1

        return (self.weights,self.biases)

    def testing(self,testData):
        correctDigits = 0
        for x,y in testData:
            if (np.argmax(self.forwardprop(x))-y)==0:
                correctDigits+=1
            if (correctDigits / 10000) > 0.95:
                for i in range(self.l-1):
                    filename1 = str(self.l) + "-layers-network-" + str(len(self.weights[i]))+"-neuron-layer" + str(i + 1) + "-weights.txt"
                    filename2 = str(self.l) + "-layers-network-" + str(len(self.biases[i])) +"-neuron-layer" + str(i + 1) + "-biases.txt"
                    np.savetxt(filename1, self.weights[i])
                    np.savetxt(filename2, self.biases[i])
        return correctDigits / 10000
    def loadParameters(self):
        weights=[]
        biases=[]
        for i in range(self.l-1):
            string1="input/" + str(self.l) + "-layers-network-" + str(len(self.weights[i]))+"-neuron-layer" + str(i + 1) + "-weights.txt"
            string2="input/" + str(self.l) + "-layers-network-" + str(len(self.biases[i])) +"-neuron-layer" + str(i + 1) + "-biases.txt"
            w=np.loadtxt(string1).reshape(self.weights[i].shape)
            b=np.loadtxt(string2).reshape(self.biases[i].shape)
            weights.append(w)
            biases.append(b)
        self.weights=weights
        self.biases=biases
    def recognition(self,newImage):
        a=newImage
        for w,b in zip(self.weights,self.biases):
            a=logi(w@a+b)
        return np.argmax(a)
    def currentAccuracy(self):
        trainingData, vd, testData = mnist_loader.load_data_wrapper()
        trainingData = list(trainingData)
        testData = list(testData)
        return self.testing(testData)

trainingData, vd, testData = mnist_loader.load_data_wrapper()
trainingData = list(trainingData)
testData = list(testData)





