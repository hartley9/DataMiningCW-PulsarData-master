#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:54:33 2019

@author: vagrant
"""

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
#####LDA#####
from scipy.linalg import pinv
import torch


#split newly sampled data and labels of new dataset
sampType = 2
if (sampType ==0):
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataOverSampled.T
elif(sampType ==1):
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataUnderSampled.T
else:
    x0,x1,x2,x3,x4,x5,x6,x7,x8 = dataSmote.T
    

data = np.vstack((x0,x1,x2,x3,x4,x5,x6,x7)).T
data = data.astype(np.float32)
labels = np.array(x8)
del x0,x1,x2,x3,x4,x5,x6,x7,x8

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


data = normalise(data)

upper = len(data)
mid = upper / 2

testGroup1 = np.random.choice(np.arange(4141), size=740, replace=False)
testGroup2 = np.random.choice(np.arange(4141,8282), size=740, replace=False)

testItems = np.concatenate([testGroup1, testGroup2], axis=None)

trainingData = np.delete(data, testItems, axis=0)
trainingLabels = np.delete(labels, testItems, axis=0)

def train(rawData, rawLabels):
    ##convert labels
    labels= []
    
    for label in rawLabels:
        if (label == 0):
            labels.append([1.0,0])
        elif(label ==1):
            labels.append([0,1.0])
            
    labels = torch.tensor(labels, dtype=torch.float32)
    data = torch.from_numpy(rawData)
    
    model = torch.nn.Sequential(torch.nn.Linear(8,3),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(3,2),
                                torch.nn.Sigmoid()
                                )
    
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    learning_rate = 0.001
    n_epochs = 2000
    
    for t in range(n_epochs):
        y_pred = model(data)
        
        loss = loss_fn(y_pred, labels)
       # print(t,loss.item())
        
        model.zero_grad()
        
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    return model

model = train(trainingData, trainingLabels)

import re
tp = tn = fp = fn = 0.0

for item in testItems:
    with torch.no_grad():
        prediction = model(torch.from_numpy(data[item,:]))
    print("Item: " + str(item))
    
    ###### split torch output to obtain metrics
    s = str(prediction)
    m = re.search(r"\[(.*?)\]", s)
    
    if m:
       m = str(m.group(1))
    else: continue
    
    m = str(m)
    neuronOuts = m.split(',')
   # print(float(neuronOuts[0]))
    print(neuronOuts)
    print(str(float(neuronOuts[0])) + '--' + str(float(neuronOuts[1])))
    if (float(neuronOuts[0]) > float(neuronOuts[1])):
        predictClass = 0
    else:
        predictClass = 1
        
    if ((int(predictClass) == 1) and (int(labels[item])) == 1):
        tp += 1
        print("correct")
    elif ((int(predictClass) == 0) and (int(labels[item])) == 0):
        tn += 1
        print("correct")
    elif ((int(predictClass) == 1) and (int(labels[item])) == 0):
        fp += 1
    elif ((int(predictClass) == 0) and (int(labels[item])) == 1):
        fn += 1
    
    print("True Class: " + str(labels[item]))
    ####################
    
print()
print()
print("metrics--------------")
accuracy = (tp+tn) / (tp+tn+fp+fn)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print("accuracy is: " + str(accuracy))
print("precision is: " + str(precision))
print("recall is: " + str(recall))
print("Time taken: " + str(time.process_time() - start))
