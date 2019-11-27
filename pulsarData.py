''''
Suggested sequence
    1) Select features
    2) Standardise
    3) Detect global outliers e.g(if <=3 or >=3)
    4) Normalise to convert so [-3,3] -> [0,1]
    5) Orthogonalise features (PCA)
    6) Cluster
    7) Detect local outliers
'''
import math
import numpy as np
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

#Normalise the data first
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

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(normalisedData[1:, 0:8])

#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

pca = PCA(n_components=5)
dataset = pca.fit_transform(data_rescaled) #new dataset

#detect global outliers

#Clustering algorithm
'''Heirarchal clustering'''

#1) proximity measures
def distance(data):
    eucData = data.copy()
    
    rows = data.shape[0]
    cols = data.shape[1]
    
    distanceMatrix = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(rows):
            
            sumTotal = 0
            
            for c in range(cols):
                
                sumTotal = sumTotal + pow((data[i,c] - data[j,c]),2)
                
            distanceMatrix[i,j] = math.sqrt(sumTotal)
        
    return distanceMatrix

distanceMatrix = distance(dataset)

#condense the matrix
condensedData = sp.spatial.distance.squareform(distanceMatrix)

#Linkages
import scipy.cluster as scClus

Z = scClus.heirarchy.linkage(condensedData)

plt.figure(figsize=(6,4))
scClus.heirarchy.dendrogram(Z)
plt.savefig("dendrogram.pdf")
                

#def heirarchalAgg():
    
#def heirarchalDiv():
    
    







