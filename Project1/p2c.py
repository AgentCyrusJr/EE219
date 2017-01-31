#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p2c Neural Networks Regressor 

Created on Wed Jan 25 16:31:01 2017

@author: huxiongfeng
"""

# Load network_backup_dataset.csv dataset from CSV URL
import pandas as pd
import numpy as np
import scipy as sp
from sklearn import  preprocessing, svm#, cross_validation
from sklearn.preprocessing import StandardScaler  
from sknn.mlp import Regressor, Layer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import cross_val_predict, cross_val_score
import pylab as pl
from sklearn.metrics import r2_score

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
# feature scaling
X = preprocessing.scale(X)
y = np.array(df[label])

print(len(X), len(y))
# neural network model
clf = Regressor(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Rectifier", units=100),
        Layer("Rectifier", units=100),
        #Layer("Rectifier", units=10),
        Layer("Linear")],
    #learning_rule ='adagrad',
    batch_size = 1,
    learning_rate=0.01,
    n_iter=10)
# 10-fold cross validation
y_predicted = cross_val_predict(clf, X, y, cv=10)
rmse = sqrt(mean_squared_error(y, y_predicted))
r2 = r2_score(y, y_predicted)
#print ("accuracy :" , accuracy)
print ("rmse: ", rmse)
print ("r2: ", r2)
# real values vs fitted values plot
pl.scatter(y_predicted,y)
pl.plot([y.min(), y.max()],[y.min(), y.max()], 'g-')
pl.xlabel('predicted')
pl.ylabel('real')
pl.show()
pl.savefig('RealValues_vs_FittedValuesPlot.png', format='png')
# residuals vs fitted values plot
y = list(y)
y_predicted = list(y_predicted)
residuals = []
for i in range(len(y)):
    residuals.append(y[i]-y_predicted[i])  
pl.scatter(y_predicted,residuals, color='blue', lw=3)
pl.plot([-0.2,1.2],[0, 0], 'g-')
pl.xlabel('predicted')
pl.ylabel('residuals')
pl.show()
pl.savefig('Residuals_vs_FittedValuesPlot', format='png')