#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p3_polynomial regression model for a fixed training and test set

Created on Sun Jan 29 16:15:09 2017

@author: huxiongfeng
"""

# Load network_backup_dataset.csv dataset from CSV URL
import pandas as pd
#import math
import numpy as np
#import scipy as sp
#import random
from sklearn import  linear_model, cross_validation#, preprocessing, svm
from sklearn.preprocessing import PolynomialFeatures
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#import tensorflow as tf
#from sklearn.preprocessing import StandardScaler  
#from sknn.mlp import Regressor, Layer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import cross_val_predict, cross_val_score
import pylab as pl
from sklearn.metrics import r2_score

#from sklearn.neural_network import MLPRegressor
#from sklearn.model_selection import cross_val_score

# load dataset from the CSV file
df = pd.read_csv('/Users/huxiongfeng/Desktop/ee219WorkSpace/network_backup_dataset.csv')
# transform textual data to numerical
# Day of Week 
df = df.replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                     [0, 1, 2, 3, 4, 5, 6]) 
df['Day of Week'] = df['Day of Week'].astype(int)
# Work-Flow-ID
df = df.replace(['work_flow_0', 'work_flow_1', 'work_flow_2', 'work_flow_3', 'work_flow_4'], 
                     [0, 1, 2, 3, 4])
df['Work-Flow-ID'] = df['Work-Flow-ID'].astype(int)
# File-name
df = df.replace(['File_0','File_1','File_2','File_3','File_4','File_5','File_6',
'File_7','File_8','File_9','File_10','File_11','File_12','File_13','File_14'
,'File_15', 'File_16', 'File_17', 'File_18' ,'File_19' ,'File_20' ,'File_21' ,
'File_22' ,'File_23' ,'File_24' ,'File_25' ,'File_26','File_27','File_28','File_29'],
['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17',
'18','19','20','21','22','23','24','25','26','27','28','29'])
df['File Name'] = df['File Name'].astype(int)

df = df[['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 
'Work-Flow-ID','File Name', 'Size of Backup (GB)', 'Backup Time (hour)']]
df['Day #'] = ((df['Week #']-1) * 7) + df['Day of Week']
     
df = df[['Day #', 'Backup Start Time - Hour of Day', 'Work-Flow-ID', 
'File Name','Size of Backup (GB)', 'Backup Time (hour)']]
         
#forecast_col = 'Size of Backup (GB)'
df.fillna(-99999, inplace=True)
label = 'Size of Backup (GB)'
# X - features
X = np.array(df.drop([label], 1))
# y - label 
y = np.array(df[label])

X = list(X)
y = list(y)

X_0 = []
X_1 = []
X_2 = []
X_3 = []
X_4 = []

y_0 = []
y_1 = []
y_2 = []
y_3 = []
y_4 = []

for i in range(len(X)):
    if X[i][2] == 0:
        X_0.append(X[i])
        y_0.append(y[i])
    if X[i][2] == 1:
        X_1.append(X[i])
        y_1.append(y[i])
    if X[i][2] == 2:
        X_2.append(X[i])
        y_2.append(y[i])
    if X[i][2] == 3:
        X_3.append(X[i])
        y_3.append(y[i])
    if X[i][2] == 4:
        X_4.append(X[i])
        y_4.append(y[i])

print(len(X_0), len(y_0))
print(len(X_1), len(y_1))
print(len(X_2), len(y_2))
print(len(X_3), len(y_3))
print(len(X_4), len(y_4))

X_0 = np.array(X_0)
X_1 = np.array(X_1)
X_2 = np.array(X_2)
X_3 = np.array(X_3)
X_4 = np.array(X_4)

y_0 = np.array(y_0)
y_1 = np.array(y_1)
y_2 = np.array(y_2)
y_3 = np.array(y_3)
y_4 = np.array(y_4)
RMSE_0 = []
RMSE_1 = []
RMSE_2 = []
RMSE_3 = []
RMSE_4 = []
for i in range(5):
# Workflow_0
    poly_0 = PolynomialFeatures(degree=i+1)
    X_0_temp = poly_0.fit_transform(X_0)
    X_0_train, X_0_test, y_0_train, y_0_test = cross_validation.train_test_split(X_0_temp, y_0, test_size = 0.1)
    clf_0 = linear_model.LinearRegression()
    clf_0.fit(X_0_train, y_0_train)
    y_0_predicted = clf_0.predict(X_0_test)
    rmse_0 = sqrt(mean_squared_error(y_0_test, y_0_predicted))
    RMSE_0.append(rmse_0)
# Workflow_1
    poly_1 = PolynomialFeatures(degree=i+1)
    X_1_temp = poly_1.fit_transform(X_1)
    X_1_train, X_1_test, y_1_train, y_1_test = cross_validation.train_test_split(X_1_temp, y_1, test_size = 0.1)
    clf_1 = linear_model.LinearRegression()
    clf_1.fit(X_1_train, y_1_train)
    y_1_predicted = clf_1.predict(X_1_test)
    rmse_1 = sqrt(mean_squared_error(y_1_test, y_1_predicted))
    RMSE_1.append(rmse_1)
# Workflow_2
    poly_2 = PolynomialFeatures(degree=i+1)
    X_2_temp = poly_2.fit_transform(X_2)
    X_2_train, X_2_test, y_2_train, y_2_test = cross_validation.train_test_split(X_2_temp, y_2, test_size = 0.1)
    clf_2 = linear_model.LinearRegression()
    clf_2.fit(X_2_train, y_2_train)
    y_2_predicted = clf_2.predict(X_2_test)
    rmse_2 = sqrt(mean_squared_error(y_2_test, y_2_predicted))
    RMSE_2.append(rmse_2)
# Workflow_3
    poly_3 = PolynomialFeatures(degree=i+1)
    X_3_temp = poly_3.fit_transform(X_3)
    X_3_train, X_3_test, y_3_train, y_3_test = cross_validation.train_test_split(X_3_temp, y_3, test_size = 0.1)
    clf_3 = linear_model.LinearRegression()
    clf_3.fit(X_3_train, y_3_train)
    y_3_predicted = clf_3.predict(X_3_test)
    rmse_3 = sqrt(mean_squared_error(y_3_test, y_3_predicted))
    RMSE_3.append(rmse_3)
# Workflow_4
    poly_4 = PolynomialFeatures(degree=i+1)
    X_4_temp = poly_4.fit_transform(X_4)
    X_4_train, X_4_test, y_4_train, y_4_test = cross_validation.train_test_split(X_4_temp, y_4, test_size = 0.1)
    clf_4 = linear_model.LinearRegression()
    clf_4.fit(X_4_train, y_4_train)
    y_4_predicted = clf_4.predict(X_4_test)
    rmse_4 = sqrt(mean_squared_error(y_4_test, y_4_predicted))
    RMSE_4.append(rmse_4)