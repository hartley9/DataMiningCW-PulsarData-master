#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:03:01 2019

@author: vagrant
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
                #print(sample)
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

#create 1d array with the label column
labels = []
for lab in range(len(data)):
  
    labels.append(data[lab,0])

#Remove label column from data set
data = np.delete(data, 0, 1)

#Label encode the label array
le = LabelEncoder()
labels = le.fit_transform(labels)



#one hot encoding of the features
tdataFrame = pd.DataFrame(data=data)
efffncoded_data = pd.get_dummies(tdataFrame)
data = pd.DataFrame.to_numpy(efffncoded_data, dtype=int, copy=False)


'''Do PCA here to reduce dimensions of data set'''

###centralise data##
'''
fulldata = np.c_[data, labels]

def centralise(data):
    centralisedData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        mu = np.mean(data[:,j])
        
        for i in range(rows):
            centralisedData[i,j] = (data[i,j]-mu)
    return centralisedData

centralisedData = centralise(fulldata)

###Start PCA data
pca = PCA(n_components=2)
pca.fit(centralisedData)
Coeff = pca.components_

transformedData = pca.transform(centralisedData)

###Plot PCA'd data
plt.figure(figsize=(12,8))
plt.plot(transformedData[:,0], transformedData[:,1], '.')

plt.xlabel("1st PC")
plt.ylabel("2nd PC")
plt.show()
'''
#randomly select 5% of the data for use in testing
testSize = int(((np.ma.size(data, axis=0))/100)*5)
splitPoint = int((np.ma.size(data, axis=0)))/2
testGroup1 = np.random.choice(np.arange(splitPoint), size=testSize, replace=False)
testGroup2 = np.random.choice(np.arange(splitPoint, (np.ma.size(data,axis=0))), size=testSize, replace=False)

testItems = np.concatenate([testGroup1, testGroup2], axis=None)

trainingData = np.delete(data, testItems, axis=0)
trainingLabels = np.delete(labels, testItems, axis=0)

testingData = []
testingLabels = []

print('--------Data splitting---------')

for item in testItems:
    testingData.append(data[int(item),:])
    testingLabels.append(labels[int(item)])



'''-----DECISION TREE----------'''
from sklearn import tree
import pydotplus
clf = tree.DecisionTreeClassifier()
#training the tree
clf = clf.fit(trainingData, trainingLabels)
#plotting the tree
tree.plot_tree(clf.fit(trainingData, trainingLabels))

#Test the tree, results are 1d array of labelling results
results = clf.predict(testingData)

'''writing pdf to filesystem
from os import system
classNames = ['poisonous', 'edible' ]
dotdata = tree.export_graphviz(clf, out_file = None, feature_names = headers[1:], class_names = classNames)
graph = pydotplus.graph_from_dot_data(dotdata)
graph.write_pdf('decisionTree.pdf')
'''

####PRINT RESULTS####
print('-------------RESULTS------------')
i = 0
tp = tn = fp = fn = 0
for item in testItems:
    item = int(item)
    predictClass = results[i]

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
    
    i += 1
    
accuracy = (tp+tn) / (tp+tn+fp+fn)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print("----------METRICS ------" )
print("accuracy is: " + str(accuracy))
print("precision is: " + str(precision))
print("recall is: " + str(recall))
#print("time taken: " + str(time.process_time() - start))


print('------------LOGISTIC REGRESSION--------------')
'''----Logistic regression----'''


