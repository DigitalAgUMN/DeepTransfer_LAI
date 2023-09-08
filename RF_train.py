# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:56:45 2022

@author: zhou1743
"""

import scipy.io
import sys
sys.path.append("../")
import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import glob,os

import pandas as pd

import pickle

data = scipy.io.loadmat(r"MODIS_train.mat")
data = data['train_data']

data = data.reshape(-1,7)
X = data[:,:-1]
Y = data[:,-1]

train_precent = np.int32(0.8*data.shape[0])
indices = np.random.permutation(data.shape[0])
X_train = X[indices[:train_precent],:]
y_train = Y[indices[:train_precent]]
X_test = X[indices[train_precent:],:]
y_test = Y[indices[train_precent:]]

rf = RandomForestRegressor(max_depth=15, random_state=0, n_jobs=8).fit(X_train, y_train)

print(rf.score(X_train, y_train))

filename = 'MODIS_LAI_RF.sav'
pickle.dump(rf, open(filename, 'wb'))

y_est = rf.predict(X_test)
print(np.mean((y_est-y_test)**2))
r = np.corrcoef(y_test, y_est)
print((r[0,1])**2)
