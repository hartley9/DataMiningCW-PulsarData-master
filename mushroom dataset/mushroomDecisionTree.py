#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:03:01 2019

@author: vagrant
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree





def loadData(filePath):
    dataRaw = []
    DataFile = open(filePath)
    missingValSamples = []
    sample = -1
    while True:
        sample += 1
        theline = DataFile.readline()
        
        if len(theline) == 0:
            break
        
        theline = theline.rstrip()
        readData = theline.split(',')
        
        for pos in range(len(readData)):
            if (readData[pos] == '?'):
                #readData[pos] = float('NaN')
                print(sample)
                missingValSamples.append(sample) #store rows with missing data
            else:
                readData[pos] = str(readData[pos])
                
        dataRaw.append(readData)
        
    DataFile.close()
    
    data = np.array(dataRaw)
    
    return data, missingValSamples

data, missingValSamples = loadData('mushroom.data')

###Drop samples with missing values from data
data = np.delete(data, missingValSamples, axis=0)

##One hot encoding of categorical data
##reading categorical data -> https://pbpython.com/categorical-encoding.html
headers = ["class",
            "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
           "gill-attachment", "gill-spacing", "gill-size", "gill-color",
           "stalk-shape", "stalk-root", 
           "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
           "stalk-color-below-ring",
           "veil-type", "veil-color", "ring-number", "ring-type",
           "spore-print-color",
           "population", 
           "habitat"];
indexes = []

for indx in range(len(data)):
    indexes.append(indx)
    

dataFrame = pd.DataFrame(data=data)
##One hot encoder
enc = LabelEncoder()
encodedData = dataFrame.apply(enc.fit_transform)
print(encodedData)

#convert dataframe back to numpy array
data = pd.DataFrame.to_numpy(encodedData, dtype=int, copy=False)
data = np.array(data[data[:,0].argsort()])

labels = []
for lab in range(len(data)):    
    labels.append(int(data[lab,0]))
   # print(data[indx,0])
    
data = np.delete(data,0,1)#drop labels from data

#randomly select 5% of the data for use in testing
testGroup1 = np.random.choice(np.arange(5644))

'''-----DECISION TREE----------'''




