import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from numpy import genfromtxt
from scipy.interpolate import *
import math
from numpy import *

# Load the dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')

# Setup before the training, using 10-fold Cross-validation
data_vol = my_data.shape[0] 
row_cut = data_vol/10*9
col_cut = 13
train_set_vol = data_vol - row_cut

# Split the data into training/testing sets
data_X_train 	= my_data[:row_cut,:col_cut]
data_X_test 	= my_data[row_cut:,:col_cut]
data_y_train 	= my_data[:row_cut,col_cut]
data_y_test 	= my_data[row_cut:,col_cut]

# color options before plot
colors = ['blue' ,'green','red']

# Create polynomial regression
for count, degree in enumerate([2, 3, 4]):
	# initial the polynomial model
    model = Pipeline([('poly', PolynomialFeatures(degree)),
					('linear', LinearRegression(fit_intercept=False))])
    # Train the model using the training sets
    poly = model.fit(data_X_train, data_y_train)
    
    # Attributes of the model
	# The coefficients
    print('Coefficients:', model.named_steps['linear'].coef_)
    # The root mean squared error
    print("Root Mean Squared Error: %.6f" % math.sqrt(np.mean((poly.predict(data_X_test) - data_y_test) ** 2)))
    # Explained variance score: 1 is perfect prediction
    print("Variance score: %.6f" % poly.score(data_X_test, data_y_test))

    # Start plot
    plt.scatter(np.arange(train_set_vol).reshape(1,train_set_vol), poly.predict(data_X_test), 
    	color=colors[count], linewidth=1, label="degree %d" % degree)
    plt.plot(np.arange(56).reshape(56,1), abs(poly.predict(data_X_test)-data_y_test), 
		color=colors[count], linewidth=1)
	

# Plot the actual points
plt.scatter(np.arange(train_set_vol).reshape(1,train_set_vol), data_y_test,  
	color='black', label='Actual Points', linewidth=2, marker = 'x')

# Add up some explanations to the figure
plt.title('Lasso with different $\\alpha$')
plt.xlabel('$n^{th}$ point')
plt.ylabel('Predicted Point & Residual')
plt.legend(loc='upper left')
plt.xticks()
plt.yticks()

plt.show()
