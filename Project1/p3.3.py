#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p3_polynomial regression model for for the average RMSE using cross validation

Created on Sun Jan 29 15:46:58 2017

@author: huxiongfeng
"""


# Load network_backup_dataset.csv dataset from CSV URL
import pandas as pd
#import math
import numpy as np
#import scipy as sp
#import random
from sklearn import  linear_model#, preprocessing, svm, cross_validation
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
    clf_0 = linear_model.LinearRegression()
    y_predicted_0 = cross_val_predict(clf_0, X_0_temp, y_0, cv=10)
    rmse_0 = sqrt(mean_squared_error(y_0, y_predicted_0))
    RMSE_0.append(rmse_0)
# Workflow_1
    poly_1 = PolynomialFeatures(degree=i+1)
    X_1_temp = poly_1.fit_transform(X_1)
    clf_1 = linear_model.LinearRegression()
    y_predicted_1 = cross_val_predict(clf_1, X_1_temp, y_1, cv=10)
    rmse_1 = sqrt(mean_squared_error(y_1, y_predicted_1))
    RMSE_1.append(rmse_1)
# Workflow_2
    poly_2 = PolynomialFeatures(degree=i+1)
    X_2_temp = poly_2.fit_transform(X_2)
    clf_2 = linear_model.LinearRegression()
    y_predicted_2 = cross_val_predict(clf_2, X_2_temp, y_2, cv=10)
    rmse_2 = sqrt(mean_squared_error(y_2, y_predicted_2))
    RMSE_2.append(rmse_0)
# Workflow_3
    poly_3 = PolynomialFeatures(degree=i+1)
    X_3_temp = poly_3.fit_transform(X_3)
    clf_3 = linear_model.LinearRegression()
    y_predicted_3 = cross_val_predict(clf_3, X_3_temp, y_3, cv=10)
    rmse_3 = sqrt(mean_squared_error(y_3, y_predicted_3))
    RMSE_3.append(rmse_3)
# Workflow_4
    poly_4 = PolynomialFeatures(degree=i+1)
    X_4_temp = poly_4.fit_transform(X_4)
    clf_4 = linear_model.LinearRegression()
    y_predicted_4 = cross_val_predict(clf_4, X_4_temp, y_4, cv=10)
    rmse_4 = sqrt(mean_squared_error(y_4, y_predicted_4))
    RMSE_4.append(rmse_4)