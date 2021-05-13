import matplotlib.pyplot as plt
import numpy as np
import time
import random
import sys

a=np.random.random_integers(0,10,200).reshape(100,2,1)
b=np.random.random_integers(10,20,200).reshape(100,2,1)
c=np.random.random_integers(20,30,200).reshape(100,2,1)
d=np.random.random_integers(30,40,200).reshape(100,2,1)
e=np.random.random_integers(30,50,200).reshape(100,2,1)
f=np.random.random_integers(50,60,200).reshape(100,2,1)
g=np.random.random_integers(60,70,200).reshape(100,2,1)
a=np.concatenate((a,b,c,d),axis=0)


def cluster(numofclusters,data):
    shape=list()
    shape.append((data.shape)[1])
    shape.append(numofclusters)
    shape.reverse()
    centroids=list(np.zeros((numofclusters,1)))

    for i in range(numofclusters):
        centroids[i]=data[random.randint(0,len(data))]
    centroids=np.array(centroids)

    clusters=list(np.zeros((len(data),1)))
    for i in range(5):
        fig, axe = plt.subplots()
        newCentroids = list()
        for i in range(numofclusters):
            newCentroids.append(list())
        for i in range(len(data)):
            temp=list()
            for j in range(len(centroids)):
                result=((data[i]-centroids[j])**2).sum()
                temp.append(result)
            temp=np.array(temp)
            clusters[i]=np.argmin(temp)
        for i in range(len(newCentroids)):
            newCentroids[i]=list(newCentroids[i])
        for i in range(len(clusters)):
            newCentroids[clusters[i]].append(data[i])


        color =["yellow","blue","orange","green","purple","cyan"]
        for i in range(len(newCentroids)):
            newCentroids[i]=np.array(newCentroids[i])
            print(len(newCentroids[i]))
            print(newCentroids[i].shape)
            axe.scatter(newCentroids[i][:, 0], newCentroids[i][:, 1], color=color[i])
        for i in range(len(newCentroids)):
            temp=newCentroids[i].sum(axis=0)/len(newCentroids[i])
            newCentroids[i]=temp

        centroids=np.array(newCentroids)
        axe.scatter(centroids[:, 0], centroids[:, 1], color="red")
        plt.show()


    return  clusters

cluster(4,a)
