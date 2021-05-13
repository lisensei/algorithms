import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data as Data
import torch.nn.utils
import torchvision.transforms as transform
import torchvision.datasets
import torchvision.models
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import math
import time
import csv
import argparse
import random
from sklearn.metrics import f1_score

with open('metrics_with_curriculum_cl.csv', 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow(["Epoch","Train Loss", "Test Loss","Train Accuracy", "Test Accuracy","Train F1", "Test F1", "Train Confusion Matrix","Test Confusition Matrix"])
#Hyper parameters

parser=argparse.ArgumentParser(description="Learning rate,noise percent and batch size")
parser.add_argument("-rn",default=3e-3)
parser.add_argument("-lr",default=3e-3)
parser.add_argument("-np",default=20)
parser.add_argument("-bs",default=16)
parser.add_argument("-mean",default=0)
parser.add_argument("-std",default=1)
parser.add_argument("-classes",default=10)
hp=parser.parse_args()
runname=hp.rn
lr=hp.lr
batchSize=hp.np
epoch=hp.bs
mean=hp.mean
std=hp.std
number_of_class=hp.classes



def addPeperNoise(source,percent):
    source = source.squeeze(dim=1)
    for image in source:
        for j in range(int(255*percent/100)):
            x = torch.randint(0, 28, (1,))
            y = torch.randint(0, 28, (1,))
            if (x + y) % 2 == 0:
                image[x, y] = 0
            else:
                image[x, y] = 255
    return source.unsqueeze(dim=1)


class TransformNoise:
    def __init__(self):
        self.percent_noise = 0.0

    def __call__(self, x):
        return addPeperNoise(x, self.percent_noise)

    def __repr__(self):
        return self.__class__.__name__ + '()'



#Data processing
curriculum_noise = TransformNoise()
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataTrain=torchvision.datasets.MNIST(root='/data',train=True,transform=transform.Compose(
        [
            transform.ToTensor(),
            curriculum_noise,
            torchvision.transforms.Normalize(mean=0.0, std=1.0)
        ]
    ),download=True)
dataTest=torchvision.datasets.MNIST(root='/data',train=False,transform=transform.Compose(
        [
            transform.ToTensor(),
            curriculum_noise,
            torchvision.transforms.Normalize(mean=mean, std=std)
        ]
    ),download=True)
trainSet=Data.DataLoader(dataset=Data.TensorDataset(dataTrain.data.to(torch.float32).unsqueeze(dim=1),
                                                    dataTrain.targets),batch_size=batchSize,shuffle=True)
testSet=Data.DataLoader(dataset=Data.TensorDataset(dataTest.data.to(torch.float32).unsqueeze(dim=1),
                                                   dataTest.targets),batch_size=batchSize,shuffle=False)



class CNN(nn.Module):
    def __init__(self, numberOfClasses):
        self.batchSize=batchSize
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.r1 = nn.ReLU();
        self.cnn2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.r2 = nn.ReLU();
        self.cnn3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.r3 = nn.ReLU();
        self.cnn4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.r4 = nn.ReLU();
        self.fc = nn.Linear(in_features=28 * 28 * 4, out_features=numberOfClasses)

    def forward(self, x):
        out = self.cnn1(x);
        out = self.r1(out)
        out = self.cnn2(out);
        z1 = self.r2(out)
        out = self.cnn3(z1 + x);
        out = self.r3(out)
        out = self.cnn4(out + z1);
        out = self.r4(out)
        out = out.reshape(self.batchSize, 28 * 28 * 4)
        out = self.fc(out)
        return out

net=CNN(number_of_class)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)

is_curriculum_enabled=False

if not is_curriculum_enabled:
    curriculum_noise.percent_noise = 50


print("Learning rate set to",lr,"Batch size set to",batchSize)
'''Training with curriculum:'''
print("\n")
print("Training with curriculum starts:")


for ite in range(epoch):
    trainLoss=0
    testLoss=0
    trainCorrect = 0
    net.batchSize=batchSize
# Train F1 variables and confusion matrix declaration
    f1ypred=torch.zeros((10,1))
    f1y=torch.zeros((10,1))
    confusionMat=torch.zeros((10,10))


    for j,(x,y) in enumerate(trainSet):
        x.to(device)
        y.to(device)
        ypred=net.forward(x)
        ypred=ypred.to(torch.float32)
        temp = torch.argmax(ypred, dim=1)
        for k,v in zip(y,temp):
            f1ypred[v]+=1
            f1y[k]+=1
            confusionMat[k,v]+=1
        trainCorrect += ((temp - y) == 0).sum()
        l = loss(ypred,y)
        trainLoss+=l
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        if j%1000==0:
            print("epoch:",ite,"batch:",j,"Loss:",l.item())
    trainAcc=trainCorrect/60000
    if is_curriculum_enabled:
        if epoch % 10 == 0:
            curriculum_noise.percent_noise += 10
#Calculate Train F1
    y1=f1y.to(torch.int32).flatten().tolist()
    y2=f1ypred.to(torch.int32).flatten().tolist()
    f1=f1_score(y1,y2,average='macro')

#Test F1 variables and confusion matrix declaration
    f1ypredTest = torch.zeros((10, 1))
    f1yTest = torch.zeros((10, 1))
    confusionMatTest=torch.zeros((10,10))
    testCorrect=0
    with torch.no_grad():
        for i in range(1):
            for j,(x,y) in enumerate(testSet):
                x.to(device)
                y.to(device)
                net.batchSize=batchSize
                ypred=net(x)
                l=loss(ypred,y)
                testLoss+=l
                temp=torch.argmax(ypred,dim=1)
                for k, v in zip(y, temp):
                    f1ypredTest[v] += 1
                    f1yTest[k] += 1
                    confusionMatTest[k, v] += 1
                testCorrect+=((temp-y)==0).sum()

    y1Test = f1yTest.to(torch.int32).flatten().tolist()
    y2Test = f1ypredTest.to(torch.int32).flatten().tolist()
    f1Test = f1_score(y1Test, y2Test, average='macro')
    testAcc=testCorrect/10000
    print(f'Train loss:{trainLoss.item():.4f}, Train accuracy:{trainAcc.item():.4f},Test loss:{testLoss.item():.4f},Test accuracy:{testAcc.item():.4f},Train F1 score:{f1:.4f},Test F1 score:{f1Test:.4f}')


    with open('metrics_with_curriculum_cl.csv', 'a', newline='') as csvfile:
        fwrite = csv.writer(csvfile, delimiter=',')
        fwrite.writerow([ite,trainLoss.item(),testLoss.item(),trainAcc.item(),testAcc.item(),f1,f1Test,confusionMat,confusionMatTest])