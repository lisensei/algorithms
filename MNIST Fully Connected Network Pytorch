import torch
import torch.nn as nn
import torchvision.transforms as transform
import torchvision.datasets

data=torchvision.datasets.MNIST(root='./data',transform=transform.ToTensor(),download=True)

X=data.data
Y=data.targets

X=X.to(torch.float32).reshape(len(X),1,784)
Y=Y.reshape(len(Y),1)

trainSet=48000
trainX=X[:trainSet]
trainY=Y[:trainSet]

testX=X[trainSet:]
testY=Y[trainSet:]

class Network(nn.Module):
    def __init__(self,inputNeurons,numberOfClasses):
        super(Network, self).__init__()
        self.l1=nn.Linear(in_features=inputNeurons,out_features=100)
        self.a1=nn.ReLU()
        self.l2=nn.Linear(in_features=100,out_features=30)
        self.a2=nn.ReLU()
        self.l3=nn.Linear(in_features=30,out_features=numberOfClasses)
    def forward(self,x):
        out=self.l1(x)
        out=self.a1(out)
        out=self.l2(out)
        out=self.a2(out)
        out=self.l3(out)
        return out


net=Network(784,10)
lr=0.001
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)

for i in range(1):
    for j,k in zip(trainX,trainY):
        ypred=net(j).reshape(1,10)

        l=loss(ypred,k)
        l.backward()

        optimizer.step()
        optimizer.zero_grad()

corret=0
with torch.no_grad():
    for i,j in zip(testX,testY):
        testy=net(i)
        if torch.argmax(testy)==j:
            corret+=1
    accuracy=corret/len(testX)

print(f'accuracy={accuracy:.4f}')
