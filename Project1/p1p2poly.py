# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from numpy import genfromtxt
from scipy.interpolate import *
import math
from numpy import *

# Load the diabetes dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')


vol_x = 450
vol_y = 13
# Split the data into training/testing sets
data_X_train = my_data[:vol_x,:vol_y]
data_X_test = my_data[vol_x:,:vol_y]

# Split the targets into training/testing sets
data_y_train = my_data[:vol_x,vol_y]
data_y_test = my_data[vol_x:,vol_y]

# set up before plot
colors = ['teal', 'yellowgreen', 'gold']
lw = 3
d = 3
#plot the test points
plt.scatter(np.arange(56).reshape(1,56), data_y_test,  color='black', label='actual')

# Create polynomial regression
for count, degree in enumerate([2, 3, 4]):
    model = Pipeline([('poly', PolynomialFeatures(degree)),
					('linear', LinearRegression(fit_intercept=False))])
    poly = model.fit(data_X_train, data_y_train)
    plt.scatter(np.arange(56).reshape(1,56), poly.predict(data_X_test), 
    	color=colors[count], linewidth=lw, label="degree %d" % degree)

plt.legend(loc='lower left')

# # The coefficients
# print('Coefficients:', regr.coef_)
# # The mean squared error
# print("Root Mean Squared Error: %.2f"
#       % math.sqrt(np.mean((model.predict(data_X_test) - data_y_test) ** 2)))
# # Explained variance score: 1 is perfect prediction
# print("Variance score: %.2f" % regr.score(data_X_test, data_y_test))

# # Plot outputs
# plt.title('Ordinary Least Square')
# plt.scatter(np.arange(56).reshape(1,56), data_y_test,  color='black', label='actual')
# plt.scatter(np.arange(56).reshape(1,56), regr.predict(data_X_test),  color='red', label='predict')
# plt.plot(np.arange(56).reshape(56,1), abs(regr.predict(data_X_test)-data_y_test), color='blue',
#          linewidth=3, label='residual')

# plt.legend(loc='upper left')
# plt.xticks([0,14,28,42,56])
# plt.yticks([0,2,4,6,8,10,20,30])

plt.show()
