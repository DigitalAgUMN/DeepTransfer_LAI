# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:12:47 2022

@author: zhou1743
"""

import scipy.io
import sys
sys.path.append("../")
import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import glob,os

import pickle
import scipy.io


filename = 'MODIS_LAI_RF.sav'
# load the model from disk
rf = pickle.load(open(filename, 'rb'))

nd = 33
nb = 7

mat = scipy.io.loadmat(r"Mead_Landsat.mat")
test_data = mat['output']
test_data[:,:-1] = test_data[:,:-1]*10000.0

X_test = test_data[:,:-1]

y_est = rf.predict(X_test)
y_est = y_est.reshape(-1,nd)
np.savetxt("RF_LAI.txt",y_est/10.0)