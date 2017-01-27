import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from sklearn import linear_model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from numpy import genfromtxt
from scipy.interpolate import *
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

# Load the dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')

# Setup before the training
col_cut = 13
my_feature = my_data[:, :col_cut]
my_target  = my_data[:, col_cut]

# color options before plot
colors = ['blue' ,'green','red', 'gold']

plt.figure(1)
# Create polynomial regression
for count, degree in enumerate([4,5,6,7]):
	# initial the polynomial model
    model = Pipeline([('poly', PolynomialFeatures(degree)),
					('linear', LinearRegression())])
    i = 0
    max_score = float('-inf')
    while i < 10 :
        data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(my_feature, my_target, test_size=0.1, random_state=random.randrange(0, 100))
        train_set_vol = len(data_X_test)
        poly = model.fit(data_X_train, data_y_train)
        score = poly.score(data_X_test, data_y_test)
        print score
        if score > max_score:
            max_score = score
            optimal_coef = model.named_steps['linear'].coef_
            best_predict = poly.predict(data_X_test)
            optimal_RMSE = math.sqrt(mean_squared_error(data_y_test, best_predict))
        i = i + 1
    # Attributes of the model
	# The coefficients
    print('Coefficients:', optimal_coef)
    # The root mean squared error
    print("Root Mean Squared Error: %.6f" % optimal_RMSE)
    # Explained variance score: 1 is perfect prediction
    print("Variance score: %.6f" % score)
    plt.scatter(data_y_test, best_predict, alpha=0.3, color=colors[count], label='degree %d' % degree, linewidth=1)
    # plt.scatter(data_y_test, best_predict-data_y_test, alpha=0.6, color=colors[count], linewidth=1)


plt.plot([data_y_test.min(), data_y_test.max()], [0, 0], 'k--', alpha=0.3, lw=1.5)    
plt.title('Residuals versus fitted values plot')
plt.xlabel('Fitted Value')
plt.ylabel('Residuals')
plt.title('Fitted values and actual values scattered plot')
plt.xlabel('Actual Value')
plt.ylabel('Fitted Value')
plt.legend(loc='upper left')
plt.show()
