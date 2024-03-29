# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import math
import time

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



start = time.process_time()
'''Classification'''
#####RBF NN#####
from scipy.linalg import pinv


#split newly sampled data and labels of new dataset
sampType = 2
if (sampType ==0):
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataOverSampled.T
elif(sampType ==1):
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataUnderSampled.T
else:
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataSmote.T
    

data = np.vstack((x0,x1,x2,x3,x4,x5,x6,x7)).T
labels = np.array(x8)
del x0,x1,x2,x3,x4,x5,x6,x7,x8

## Global configs
nPrototypes = 20
nClasses = 2

def maxDist(m1,m2):
    maxDist = -1

    for i in range(len(m1)):
        for j in range(len(m2)):            
            distance = dist(m1[i,:],m2[j,:])

            if (distance > maxDist):
                maxDist = distance

    return maxDist

def RBFTrain(data,labels):
    ## Converting labels
    convLabels = []

    for label in labels:
        if (label == 0):
            convLabels.append([1,0])
        elif (label == 1):
            convLabels.append([0,1])
    
    ## Generating prototypes
    group1 = np.random.randint(0,4042,size=nPrototypes)
    group2 = np.random.randint(4042,8167,size=nPrototypes)

    prototypes = np.vstack([data[group1,:],data[group2,:]])

    ## Calculating Sigma
    distance = maxDist(prototypes,prototypes)
    sigma = distance/math.sqrt(nPrototypes*nClasses)

    ## For each item in training set, get the output
    dataRows = data.shape[0]
    
    output = np.zeros(shape=(dataRows,nPrototypes*nClasses))

    for item in range(dataRows):
        out = []

        for proto in prototypes:
            distance = dist(data[item], proto)
            neuronOut = np.exp(-distance/np.square(sigma))
            out.append(neuronOut)

        output[item,:] = np.array(out)

    weights = np.dot(pinv(output), convLabels)

    return weights, prototypes, sigma
    
def RBFPredict(item, prototypes, weights,sigma):
    out = []

    ## Hidden layer
    for proto in prototypes:
        distance = dist(item,proto)
        neuronOut = np.exp(-(distance)/np.square(sigma))
        out.append(neuronOut)
        
    netOut = []
    for c in range(nClasses):
        result = np.dot(weights[:,c],out)
        netOut.append(result)

    return np.argmax(netOut)

data = normalise(data)
# if under sampled then size =15 (33) AND (33,66)
# if over or smote then size = 100 ()
#
testGroup1 = np.random.choice(np.arange(4142),size=100,replace=False)
testGroup2 = np.random.choice(np.arange(4142,8282),size=15,replace=False)

testItems = np.concatenate([testGroup1, testGroup2],axis=None)
print(testItems)

#for training data delete the test items from array
trainingData = np.delete(data,testItems,axis=0)
trainingLabels = np.delete(labels,testItems,axis=0)

weights, prototypes, sigma = RBFTrain(trainingData,trainingLabels)

## Prediction and accuracy
tp = tn = fp = fn = 0
for item in testItems:
    predictClass = RBFPredict(data[item,:],prototypes,weights,sigma)

    print("Item: " + str(item))
    print("Predicted Class: " + str(predictClass))
    print("True Class: " + str(labels[item]))

    if ((int(predictClass) == 1) and (int(labels[item])) == 1):
        tp += 1
    elif ((int(predictClass) == 0) and (int(labels[item])) == 0):
        tn += 1
    elif ((int(predictClass) == 1) and (int(labels[item])) == 0):
        fp += 1
    elif ((int(predictClass) == 0) and (int(labels[item])) == 1):
        fn += 1

accuracy = (tp+tn) / (tp+tn+fp+fn)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print("metrics ------" )
print("accuracy is: " + str(accuracy))
print("precision is: " + str(precision))
print("recall is: " + str(recall))
print("time taken: " + str(time.process_time() - start))




