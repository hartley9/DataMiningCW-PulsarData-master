#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:41:59 2019

@author: vagrant
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import math

def readData(filePath):
    
    dataRaw = []
    labelsRaw = []
    fullData = []
    DataFile = open(filePath)
    
    
    while True: 
        theline = DataFile.readline()
        
        if len(theline) == 0:
            break
        
        theline = theline.rstrip()
        readData = theline.split(",")
        
        for pos in range(len(readData)-1):
            readData[pos] = (readData[pos]);
            
            
        if (readData[0] == 'M'):
            readData[0] = '0'
        if (readData[0] == 'F'):
            readData[0] = '1'
        if (readData[0] == 'I'):
            readData[0] = '2'
            
        dataRaw.append(readData[0:8])

        if (readData[8] == ' positive'):
            labelsRaw.append(1)
        if (readData[8] == 'negative'):
            labelsRaw.append(0)

            
    DataFile.close()
    
    data = np.array(dataRaw)
    labels = np.array(labelsRaw)
    data = data.astype(float)
    
    return data, labels


data, labels = readData('abalone19.csv') 

concatData = np.c_[data,labels] #this data will be oversampled

#sort data according to class to integrate with RBFNN functions
concatData = np.array(concatData[concatData[:,8].argsort()])




'''Pre-processing'''
def normalise(data):
    normalisedData = data.copy()

    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        maxElement = np.amax(data[:,j])
        minElement = np.amin(data[:,j])

        for i in range(rows):
            normalisedData[i,j] = (data[i,j] - minElement) / (maxElement - minElement)

    return normalisedData


'''
imbalanced data methods
'''    

def overSample(data, k):
    data = data.copy()
    
    labelLPositions = []
    
    #first find all indicies with label 1
    for rowPos in range(len(data)):
        if(data[rowPos,8] == 1):
            labelLPositions.append(rowPos);
    
   
    #select (with replacement) k indicies to duplicate
    for n in range(k):
        selectedItemIndex = labelLPositions[random.randint(0,len(labelLPositions)-1)]
        selectedItem = data[selectedItemIndex]
        data = np.vstack((data, selectedItem));

        
    return data


dataOverSampled = overSample(concatData, 4108)

'''undersampling'''
def underSample(data, k):
    newData = np.array([])
    
    labelLPositions = []
    
    #first find all indicies with label 1
    for rowPos in range(len(data)):
        if (data[rowPos,8] == 0):
            labelLPositions.append(rowPos)
            
    #select k indicies to eliminate
    toEliminate = random.sample(labelLPositions, k)
    
    for rowPos in range(len(data)):
        if(rowPos not in toEliminate):
            if (len(newData) == 0):
                newData = data[rowPos,:].copy()
            else:
                newData = np.vstack((newData, data[rowPos,:]))
                
    return newData

dataUnderSampled = underSample(concatData, 4108) # make classes balanced

'''SMOTE'''
def dist(p1, p2):
    sumTotal = 0

    for c in range(len(p1)):
        sumTotal = sumTotal + pow((p1[c] - p2[c]),2)

    return math.sqrt(sumTotal)
    
def smote(data, n):
    newData = data.copy()
    
    labelLPositions = []
    
    #find all indicies with label 1
    for rowPos in range(len(data)):
        if (data[rowPos,8] == 1):
            labelLPositions.append(rowPos)
    
    for interation in range(n):
        #randomly choose an item with label 1
        itemI = labelLPositions[random.randint(0,len(labelLPositions)-1)]
        
        #get k nearest neighbours ***HERE K=2***
        dists = []
        indexDists = []
        
        for item in labelLPositions:
            if (item == itemI):
                continue
            
            dists.append(dist(data[item], data[itemI]))
            indexDists.append(item)
        
        kNeighbours = []
        
        for neigh in range(2):
            nearest = np.argmin(dists)
            kNeighbours.append(indexDists[nearest])
            dists[nearest] = float("inf")
        
        #randomly choose one of k neighbours
        itemJ = kNeighbours[random.randint(0,len(kNeighbours)-1)]
        
        #get vector between two points, but for 8 features
        x0 = data[itemJ,0] - data[itemI,0]
        x1 = data[itemJ,1] - data[itemI,1]
        x2 = data[itemJ,2] - data[itemI,2]
        x3 = data[itemJ,3] - data[itemI,3]
        x4 = data[itemJ,4] - data[itemI,4]
        x5 = data[itemJ,5] - data[itemI,5]
        x6 = data[itemJ,6] - data[itemI,6]
        x7 = data[itemJ,7] - data[itemI,7]
        
        alpha = random.random()
        
        newPoint = [data[itemI,0] + alpha*x0, 
                    data[itemI,1] + alpha*x1, 
                    data[itemI,2] + alpha*x2, 
                    data[itemI,3] + alpha*x3, 
                    data[itemI,4] + alpha*x4,
                    data[itemI,5] + alpha*x5,
                    data[itemI,6] + alpha*x6,
                    data[itemI,7] + alpha*x7, 1
                    ]
        newData = np.vstack((newData,newPoint))
    
    return newData
        
dataSmote = smote(concatData, 4108)        




'''Classification'''
#####LDA#####
from scipy.linalg import pinv


#split newly sampled data and labels of new dataset
sampType = 1
if (sampType ==0):
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataOverSampled.T
elif(sampType ==1):
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataUnderSampled.T
else:
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataSmote.T
    

data = np.vstack((x0,x1,x2,x3,x4,x5,x6,x7)).T
labels = np.array(x8)
del x0,x1,x2,x3,x4,x5,x6,x7,x8


####start of lda####
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def meanFeatures(m1):
    mean = np.zeros(m1.shape[1])

    for c in range(m1.shape[1]):
        sumColumn = 0
        for r in range(m1.shape[0]):
            sumColumn = sumColumn + m1[r][c]

        mean[c] = sumColumn/m1.shape[0]

    return mean
        
data = data[0:66,:]
labels = labels[0:66]

testGroup1 = np.random.choice(np.arange(33),size=5,replace=False)
testGroup2 = np.random.choice(np.arange(33,66),size=5,replace=False)

testItems = np.concatenate([testGroup1, testGroup2],axis=None)
testData = data[testItems,:]

trainingData = np.delete(data,testItems,axis=0)
trainingLabels = np.delete(labels,testItems,axis=0)

## 1st step: get mean of features
means = np.zeros((2,trainingData.shape[1]))

means[0,:] = meanFeatures(trainingData[0:27,:])
means[1,:] = meanFeatures(trainingData[27:56,:])

## For the scatter matrices, we will need the overall mean
overallMean = meanFeatures(trainingData)

## 2nd step: let's calculate the with-in class scatter matrix:
SW = np.zeros((trainingData.shape[1],trainingData.shape[1]))

# First class:
for p in range(0,27):
    diff = (trainingData[p,:] - means[0,:]).reshape(trainingData.shape[1],1)
    
    SW += diff.dot(diff.T)

# Second class:
for p in range(27,56):
    diff = (trainingData[p,:] - means[1,:]).reshape(trainingData.shape[1],1)
    
    SW += diff.dot(diff.T)

## 3rd step: now let's calculate the between-class scatter matrix
SB = np.zeros((data.shape[1],data.shape[1]))

for c in range(2):
    diff = (means[c,:] - overallMean).reshape(trainingData.shape[1],1)
    
    SB += 28 * diff.dot(diff.T)

## 4th step: Calculate eigen-values and eigen-vectors
invSW = linalg.inv(SW)
eigVals, eigVectors = linalg.eig(invSW.dot(SB))

## 5th step: Find top k eigen-vectors:
orderedEigVectors = np.empty(eigVectors.shape)

tmp = eigVals.copy()

maxValue = float("-inf")
maxValuePos = -1

for i in range(len(eigVectors)):

    maxValue = float("-inf")
    maxValuePos = -1
        
    for n in range(len(eigVectors)):
        if (tmp[n] > maxValue):
            maxValue = tmp[n]
            maxValuePos = n

    orderedEigVectors[:,i] = eigVectors[:,maxValuePos]
    tmp[maxValuePos] = float("-inf")

k = 2
projectionMatrix = orderedEigVectors[:,0:k]

## 6th Step: Project the dataset
ldaData = trainingData.dot(projectionMatrix)
testLDA = testData.dot(projectionMatrix)

plt.figure(figsize=(6,4))

plt.plot(ldaData[0:27,0],ldaData[0:27,1],"r.")
plt.plot(ldaData[27:56,0],ldaData[27:56,1],"g.")

plt.plot(testLDA[0:5,0],testLDA[0:5,1],"rx")
plt.plot(testLDA[5:10,0],testLDA[5:10,1],"gx")

plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

plt.savefig("lda.pdf")

plt.close()

## Classification
k = 1
projectionMatrix = orderedEigVectors[:,0:k]

ldaData = trainingData.dot(projectionMatrix)
testLDA = testData.dot(projectionMatrix)
threshold = overallMean.dot(projectionMatrix)

plt.figure(figsize=(6,4))

plt.plot(ldaData[0:27,0],np.zeros(27),"r.")
plt.plot(ldaData[27:56,0],np.zeros(27),"g.")

plt.plot(testLDA[0:5,0],np.zeros(5),"rx",markersize=12)
plt.plot(testLDA[5:10,0],np.zeros(5),"gx",markersize=12)

plt.plot(threshold,0,"o")

plt.xlabel("1st Principal Component")

plt.savefig("lda1D.pdf")

plt.close()

