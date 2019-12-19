import re
import math
import numpy as np
import scipy.cluster as sc
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sklearn
import time


def readData(path):
    '''
    Specialized readData for mushrom set
    Returns the set and flagged indices (where data missing)
    '''
    doublelist = []
    flagged = []
    datafile = open(path, "r")
    lines = datafile.readlines()
    dictionaries = []  # dynamically allocate characters to numbers
    cols = range(len(lines[0].split(',')))
    for i in cols:
        dictionaries.append({})
    for row in range(len(lines)):
        entrylist = []
        entries = lines[row].split(',')
        for col in cols:
            if entries[col] == '?':
                entrylist.append(float('NaN'))
                flagged.append(row)
            else:
                if entries[col] not in dictionaries[col]:
                    dictionaries[col][entries[col]] = len(dictionaries[col])
                entrylist.append(dictionaries[col][entries[col]])
        doublelist.append(entrylist)
    datafile.close()
    return doublelist, flagged


def dropMising(data, flagged):
    return np.delete(data, flagged, axis=0)


def getMedian(vector):
    newVector = np.array([])

    for item in vector:
        if (not np.isnan(item)):
            newVector = np.append(newVector, item)

    return np.median(newVector)


def replace(vector, value):
    newVector = vector.copy()

    for pos in range(len(vector)):
        if (np.isnan(vector[pos])):
            newVector[pos] = value

    return newVector


def replaceByMedian(data):
    newData = data.copy()

    for col in range(len(data[0, :])):
        for row in range(len(data)):
            if (np.isnan(data[row, col])):
                median = getMedian(data[:, col])
                newData[:, col] = replace(data[:, col], median)
                break

    return newData


def meanFeatures(m1):
    return np.sum(m1, axis=0) / m1.shape[0]

def main():
    mushrom, flagged = np.array(readData("mushroom.data"))
    mushrom = dropMising(mushrom, flagged)

    mushrom_LABELS = mushrom[:, 0]
    mushrom_set = mushrom[:, 1:]

    # mushrom_set = normalize(mushrom_set)
    # nomissing data test groups
    mushrom_rows = mushrom_set.shape[0]
    # last 5% for each group will be testing data rest will be training, for each class 5% (edible, poison)
    mushrom_testgroup1 = np.random.choice(np.arange(int(
        mushrom_rows/2)), size=int(mushrom_rows*0.05), replace=False)
    mushrom_testgroup2 = np.random.choice(np.arange(int(
        mushrom_rows/2), mushrom_rows), size=int(mushrom_rows*0.05), replace=False)
    mushrom_tests_indeces = np.concatenate(
        [mushrom_testgroup1, mushrom_testgroup2], axis=None)

    mushrom_tests_set = mushrom_set[mushrom_tests_indeces, :]

    # update everything
    mushrom_set = np.delete(
        mushrom_set, mushrom_tests_indeces, axis=0)
    mushrom_LABELS = np.delete(
        mushrom_LABELS, mushrom_tests_indeces, axis=0)
    mushrom_rows = mushrom_set.shape[0]

    # mean of features
    mushrom_means = np.zeros((2, mushrom_set.shape[1]))

    mushrom_means[0, :] = meanFeatures(
        mushrom_set[:int(mushrom_rows/2), :])
    mushrom_means[1, :] = meanFeatures(
        mushrom_set[int(mushrom_rows/2):, :])

    mushrom_totalMean = meanFeatures(mushrom_set)

    mushrom_scattermatrix = np.zeros(
        (mushrom_set.shape[1], mushrom_set.shape[1]))

    # First class :
    for p in range(0, int(mushrom_rows/2)):
        diff = (mushrom_set[p, :] - mushrom_means[0, :]
                ).reshape(mushrom_set.shape[1], 1)

        mushrom_scattermatrix += diff.dot(diff.T)

    # Second class :
    for p in range(int(mushrom_rows/2), mushrom_rows):
        diff = (mushrom_set[p, :] - mushrom_means[1, :]
                ).reshape(mushrom_set.shape[1], 1)

        mushrom_scattermatrix += diff.dot(diff.T)

    SB = 0

    for c in range(2):
        diff = (mushrom_means[c, :] - mushrom_totalMean).reshape(
            mushrom_set.shape[1], 1)

        SB += int(mushrom_rows/2) * diff.dot(diff.T)

    invSW = sp.linalg.inv(mushrom_scattermatrix)
    eigVals, eigVectors = sp.linalg.eig(invSW.dot(SB))

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

    orderedEigVectors[:, i] = eigVectors[:, maxValuePos]
    tmp[maxValuePos] = float("-inf")
    projectionMatrix = orderedEigVectors[:, :2]  # 2 kclusters
    ldaData = mushrom_set.dot(projectionMatrix)
    testLDA = mushrom_tests_set.dot(projectionMatrix)

    plt.figure(figsize=(6, 4))

    plt.plot(ldaData[0:45, 0], ldaData[0:45, 1], "r.")
    plt.plot(ldaData[45:90, 0], ldaData[45:90, 1], "g.")

    plt.plot(testLDA[0:5, 0], testLDA[0:5, 1], "rx")
    plt.plot(testLDA[5:10, 0], testLDA[5:10, 1], "gx")

    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")

    plt.savefig("lda.pdf")

    # testset = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(testset[int(testset.shape[0]/2):])


main()
