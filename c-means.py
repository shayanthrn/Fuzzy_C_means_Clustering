import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cost(memberships,values,centroids,m):
    res=0
    for j in range(len(values)):
        for i in range(len(centroids)):
            res += (memberships[i][j]**m)*distance(values[j], centroids[i])
    return res

def calculate_centroids(memberships,values,centroids,m):
    for i in range(len(centroids)):
        uppertemp = [0 for j in range(len(values[0]))]
        lowertemp = 0
        for k in range(len(values)):
            uppertemp += (memberships[i][k]**m)*values[k]
            lowertemp += (memberships[i][k]**m)
        centroids[i]= uppertemp/lowertemp

def distance(A,B):
    temp=0
    for i in range(len(A)):
        temp += (A[i]-B[i])**2
    return temp**0.5


def calculatememberships(memberships,values,centroids,m):
    for i in range(len(memberships)):
        for k in range(len(values)):
            temp = 0
            for j in range(len(centroids)):
                temp += (distance(values[k],centroids[i])/distance(values[k],centroids[j]))**(2/(m-1))
            memberships[i][k]= 1/temp



def clustering_cmeans(dataset,C,m):

    values=np.array(dataset.values)

    mins=np.amin(values, axis=0) #minimums of cols 
    maxs=np.amax(values, axis=0) #maximum of cols

    #initial centroids
    centroids=[]
    for i in range(C):
        point=[]
        for j in range(len(values[0])):
            point.append(np.random.uniform(mins[j],maxs[j]))
        centroids.append(point)

    memberships = np.zeros((C,len(values)))
    calculatememberships(memberships,values,centroids,m)
    costs=0
    for i in range(100):
        calculate_centroids(memberships,values,centroids,m)
        calculatememberships(memberships,values,centroids,m)
        costs=cost(memberships,values,centroids,m)
    return costs











dataset = pd.read_csv("data3.csv",header=None)
costs=[]
for i in range(1,7):
    print(i)
    costs.append(clustering_cmeans(dataset,i,2))
plt.plot(costs)
plt.show()