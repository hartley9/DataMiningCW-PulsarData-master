#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:40:48 2019

@author: kobi
"""

''''
Suggested sequence
    1) Select features
    2) Standardise
    3) Detect global outliers e.g(if <=3 or >=3)
    4) Normalise to convert so [-3,3] -> [0,1]
    5) Orthogonalise features (PCA)
    6) Cluster
    7) Detect local outliers

htru2 data
1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class 

'''
import math
import numpy as np
import scipy.cluster as sc
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

print("Initialising...")
rawPulsarData = [] #Intialise list to hold raw pulsar data
DataFile = open("HTRU_2.csv", "r") #Open csv file

#Split each item of data on comma (given csv) and append to list
while True:
    theline = DataFile.readline()
    if len(theline) == 0:
        break
    readData = theline.split(",")
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos])
    rawPulsarData.append(readData)

DataFile.close()
data = np.array(rawPulsarData)
#data = origionalData[:, 0:8]

def normalise(data):
    normalisedData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        maxEl = np.amax(data[:,j])
        minEl = np.amin(data[:,j])
        
        for i in range(rows):
            normalisedData[i,j]=(data[i,j]-minEl)/(maxEl-minEl)
    return normalisedData

normalisedData = normalise(data)


'''PCA'''
'''centralise data'''
def centralise(data):
    centralisedData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        mu = np.mean(data[:,j])
        
        for i in range(rows):
            centralisedData[i,j] = (data[i,j]-mu)
    return centralisedData

centralisedData = centralise(normalisedData)

'''Correlation matrix'''
corrMatrix = np.corrcoef(centralisedData, rowvar=False)


pca = PCA(n_components=5)
pca.fit(centralisedData)
Coeff = pca.components_

transformedData = pca.transform(centralisedData)

plt.figure(figsize=(12,8))

plt.plot(transformedData[:,0], transformedData[:,1], '.')

plt.xlabel("1st PC")
plt.ylabel("2nd PC")
plt.show()

'''
pca = PCA(n_components=2)
pca.fit(centralisedData)
tranformedData = pca.transform(centralisedData)
'''

'''k-means clustering'''


def standard(data):
    standardData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    for j in range(cols):
        sigma = np.std(data[:,j])
        mu = np.mean(data[:,j])
        
        for i in range(rows):
            standardData[i,j] = (data[i,j] - mu) / sigma
            
    return standardData

standardData = standard(transformedData)
centroids, distortion = sc.vq.kmeans(standardData, 2)


def dist(p1, p2):
    sumTotal = 0
    
    for c in range(len(p1)):
        sumTotal = sumTotal + pow((p1[c] - p2[c]), 2)
    return math.sqrt(sumTotal)

group1 = np.array([])
group2 = np.array([])

for d in standardData:
    if (dist(d, centroids[0,:]) < dist(d, centroids[1,:])):
        if (len(group1)==0):
            group1 = d
        else:
            group1 = np.vstack((group1,d))
    else:
        if(len(group2)==0):
            group2 = d
        else:
            group2 = np.vstack((group2, d))
            
plt.figure(figsize=(12,8))

plt.plot(group1[:,0], group1[:,1],'r')
plt.plot(group2[:,0], group2[:,1], 'g')

plt.plot(centroids[0,0], centroids[0,1], 'rx')
plt.plot(centroids[1,0], centroids[1,1], 'gx')

plt.show()

