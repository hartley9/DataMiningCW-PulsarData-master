#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:03:01 2019

@author: vagrant
"""

import numpy as np
import pandas as pd

##reading categorical data -> https://pbpython.com/categorical-encoding.html
headers = ["cap-shape", "cap-surface", "cap-color", "bruises", "odor",
           "gill-attachment", "gill-spacing", "gill-size", "gill-color",
           "stalk-shape", "stalk-root", 
           "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
           "stalk-color-below-ring",
           "veil-type", "veil-color", "ring-number", "ring-type",
           "spore-print-color",
           "population", 
           "habitat"];

df = pd.read_csv('mushroom.data', header=None, names=headers, na_values="?")


def loadData(filePath):
    dataRaw = []
    DataFile = open(filePath)
    
    while True:
        theline = DataFile.readline()
        
        if len(theline) == 0:
            break
        
        theline = theline.rstrip()
        readData = theline.split(',')
        
        for pos in range(len(readData)):
            if (readData[pos] == '?'):
                readData[pos] = float('NaN')
                print('missing')
            else:
                readData[pos] = float(readData[pos])
                
        dataRaw.append(readData)
    DataFile.close()
    
    data = np.array(dataRaw)
    
    return data

data = loadData('mushroom.data')