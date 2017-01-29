import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import random

# Load the dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')
col_cut = 13

# Here we train the model with all the features
my_feature = my_data[:, :col_cut]
my_target  = my_data[:, col_cut]

i = 0
max_score = 0
while i < 10 :
	data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(my_feature, my_target,  random_state=random.randrange(0, 100), test_size=0.1)
	train_set_vol = len(data_X_test)
	# Create linear regression object
	linear = linear_model.LinearRegression()
	# Train the model using the training sets
	linear.fit(data_X_train, data_y_train)
	score, permutation_scores, pvalue = permutation_test_score(linear, data_X_train, data_y_train, cv=10, n_jobs=1)
	print score, permutation_scores, pvalue
	if linear.score(data_X_test, data_y_test) > max_score:
		max_score = linear.score(data_X_test, data_y_test)
		optimal_coef = linear.coef_
		best_predict = linear.predict(data_X_test)
		optimal_RMSE = math.sqrt(mean_squared_error(data_X_test, data_y_test))
		optimal_score= max_score	
	i = i + 1
# The coefficients
print('Coefficients:', optimal_coef)
# The root mean squared error
print("Root Mean Squared Error: %.3f" % optimal_RMSE)
# Explained variance score: 1 is perfect prediction
print("Variance score: %.3f" % optimal_score)
print "best r2_score:", r2_score(data_y_test, best_predict)

# Add up some explanations to the figure
plt.figure(1)
plt.scatter(data_y_test, best_predict, alpha=0.6, color='red', linewidth=1)
plt.plot([data_y_test.min(), data_y_test.max()], [data_y_test.min(), data_y_test.max()], 'k--', lw=4)
plt.title('Fitted values and actual values scattered plot')
plt.xlabel('Actual Value')
plt.ylabel('Fitted Value')

plt.figure(2)
plt.scatter(data_y_test, best_predict-data_y_test, alpha=0.6, color='blue', linewidth=1)
plt.plot([data_y_test.min(), data_y_test.max()], [0, 0], 'k--', lw=4)
plt.title('Residuals versus fitted values plot')
plt.xlabel('Fitted Value')
plt.ylabel('Residuals')

plt.show()
