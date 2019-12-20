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


def readCSV(path):
    doublelist = []
    # pattern = re.compile("-?\d+\.*\d*")
    csvfile = open(path, "r")
    lines = csvfile.readlines()
    for line in lines:
        entrylist = []
        for entry in line.split(','):
            # obj = pattern.search(entry)
            # if obj != None:
            entrylist.append(float(entry))
            # else:
            #     entrylist.append(entry)
        doublelist.append(entrylist)
    csvfile.close()
    return doublelist


def normalize(data):
    '''
    Normalize data
    '''
    normal = np.zeros(data.shape)
    normal = (data - np.amin(data, 0))/(np.amax(data, 0) - np.amin(data, 0))
    return normal


def centralize(data):
    '''
    Centralize data
    '''
    central = np.zeros(data.shape)
    central = data - np.mean(data, 0)
    return central


def pearsonCorr(data):
    '''
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    '''
    corcoef = np.cov(data, None, False)
    try:
        diagonal = np.diag(corcoef)
    except ValueError:
        return corcoef/corcoef
    sd = np.sqrt(diagonal.real)  # only real values in diagonal
    corcoef /= sd[:, None]  # the other way round lol
    corcoef /= sd[None, :]

    np.clip(corcoef, -1, 1, out=corcoef)  # clip to -1,1

    return corcoef


def standardize(data):
    '''
    Standardize data
    '''
    standard = np.zeros(data.shape)
    standard = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return standard


def dist(p1, p2):
    return math.sqrt(np.sum(np.power(p1-p2, 2)))


def euclidDistance(data):
    '''
    Euclid distance calculator
    '''
    rows = data.shape[0]
    cols = data.shape[1]
    distanceMatrix = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(rows):

            sumTotal = 0

            for c in range(cols):
                sumTotal = sumTotal + pow((data[i, c] - data[j, c]), 2)

            distanceMatrix[i, j] = math.sqrt(sumTotal)

    return distanceMatrix


def euclidVectorizedDistance(data):
    # may be not :(
    distanceMatrix = -2 * np.dot(data, data[:, np.newaxis]) + np.dot(data, data[:, np.newaxis]) + np.dot(data, data[:, np.newaxis])
    return distanceMatrix

def accuracy(trainedLabels, actualLabels):
    rows = actualLabels.shape[0]
    accuracy = 0
    for row in range(rows):
        if trainedLabels[row] == actualLabels[row]:
            accuracy += 1
    return accuracy / rows

def main():

    HTRU = np.array(readCSV("HTRU_2.csv"))
    HTRU_LABELS = HTRU[:, -1]
    HTRU_set = HTRU[:, 0:8]  # drop labels
    # preprocess data
    HTRU_normal = normalize(HTRU_set)
    HTRU_centralized = centralize(HTRU_normal)
    corrMatrix = pearsonCorr(HTRU_centralized)

    # pca transform data (reduce dimensions)
    pca = PCA(n_components=5)
    pca.fit(HTRU_centralized)  # Call 'fit' with appropriate arguments
    HTRU_pca = pca.transform(HTRU_centralized)

    # plot the thing
    plt.figure("PCA-processed data", figsize=(12, 8))
    plt.title("PCA-processed data")
    plt.plot(HTRU_pca[:, 0], HTRU_pca[:, 1], '.')
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.savefig("HTRU PCA processed.pdf")
    plt.show(block=False)  # display initial pca-processed data

    # k-means
    HTRU_standard = standardize(HTRU_pca)
    codebook, distortion = sc.vq.kmeans(HTRU_standard, 2)

    cluster1 = np.zeros(HTRU_standard.shape)
    cluster2 = np.zeros(HTRU_standard.shape)

    size1 = 0
    size2 = 0
    rows = range(HTRU_standard.shape[0])

    kmeansLabels = np.zeros(HTRU_standard.shape[0])

    for row in rows:
        # compare first 2 features(codeb[0,] and codeb[1,]) and cluster them into 2 clusters
        if (dist(HTRU_standard[row], codebook[0, :]) < dist(HTRU_standard[row], codebook[1, :])):
            cluster1[size1] = HTRU_standard[row]
            size1 += 1
        else:
            kmeansLabels[row] = 1
            cluster2[size2] = HTRU_standard[row]
            size2 += 1

    # this is the fastest when vectorization is difficult/not possible
    # operations on the fully created array is instant almost
    # whole array is stored in memmory and then trimmed at the end
    cluster1 = cluster1[0:size1]  # drop leftovers
    cluster2 = cluster2[0:size2]  # drop leftovers

    # show what it looks like
    plt.figure("K-Means Clustering", figsize=(12, 8))
    plt.title("K-Means Clustering")
    plt.plot(cluster1[:, 0], cluster1[:, 1], 'b.')
    plt.plot(cluster2[:, 0], cluster2[:, 1], 'g.')

    plt.plot(codebook[0, 0], codebook[0, 1], 'rx')
    plt.plot(codebook[1, 0], codebook[1, 1], 'rx')
    plt.savefig("HTRU kmeans clustering.pdf")
    plt.show(block=False)

    # Hierarcial
    HTRU_distances = sklearn.metrics.pairwise.euclidean_distances(HTRU_pca)
    HTRU_noDiag = sp.spatial.distance.squareform(HTRU_distances, checks=False)

    # linkages
    linkage = sc.hierarchy.linkage(HTRU_noDiag)
    plt.figure("Dendogram level relation of 20 clusters", figsize=(10,4))
    plt.title("Dendogram level relation of 20 clusters")
    sc.hierarchy.dendrogram(linkage, truncate_mode='level', p=20) # 20 clusters truncation
    plt.savefig("HTRU dendogram 20 clusters.pdf")
    plt.show(block=False)

    plt.figure("Dendogram lastp, with 20 last clusters", figsize=(10,4))
    plt.title("Dendogram lastp, with 20 last clusters")
    sc.hierarchy.dendrogram(linkage, truncate_mode='lastp', p=20)
    plt.savefig("HTRU 20 truncated dendogram.pdf")
    plt.show(block=False)
    

    # Finally do metrics
    # Simplest metric - accuracy of results compared to labels
    print(accuracy(kmeansLabels, HTRU_LABELS))
    input("Press [Enter] to exit")

main()
