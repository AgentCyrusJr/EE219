import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import random

# Load the dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')

# Setup before the training
col_cut = 13
my_feature = my_data[:, :col_cut]
my_target  = my_data[:, col_cut]

# training begins
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(my_feature, my_target, test_size=0.1, random_state=random.randrange(0, 100))
train_set_vol = len(data_X_test)
tuningAlpha = [0.1, 0.01, 0.001]
lasso = LassoCV(normalize=True, alphas=tuningAlpha, cv=10)
lasso.fit(data_X_train,data_y_train)
prediction = lasso.predict(data_X_test)	
print "optimal alpha: ", lasso.alpha_
print "optimal coefficients: ", lasso.coef_
print "best RMSE via 10-fold cross validation: %.3f " %math.sqrt(np.mean((prediction - data_y_test) ** 2))

# Plot the required figure
plt.scatter(np.arange(train_set_vol).reshape(1,train_set_vol), prediction, 
	color='red', linewidth=1, label="alpha %.3f" % lasso.alpha_)
plt.plot(np.arange(train_set_vol).reshape(train_set_vol,1), abs(prediction-data_y_test), 
	color='blue', linewidth=1)

plt.scatter(np.arange(train_set_vol).reshape(1,train_set_vol), data_y_test, 
	color='black', label='Actual Value', linewidth=2, marker = 'x')

# Add up some explanations to the figure
plt.title('Lasso with different $\\alpha$')
plt.xlabel('Actual Value / $n^{th}$ point')
plt.ylabel('Predicted Value / Residual')
plt.legend(loc='upper left')
plt.xticks([0,14,28,42,56])
plt.yticks([0, 3, 6, 9, 12, 15, 30, 45, 60, 75])

plt.show()
