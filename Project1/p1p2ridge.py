import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from numpy import genfromtxt

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
colors = ['teal', 'yellowgreen', 'gold', 'red']

# Create Ridge regression object with different alpha
for count, alpha in enumerate([1, 0.1, 0.01, 0.001]):
	# initial the Ridge Regression
	rid = linear_model.Ridge(alpha, fit_intercept = False)
	# Train the model using the training sets
	rid.fit(data_X_train, data_y_train)

	# Attributes of the model
	# The coefficients
	print('Coefficients:', rid.coef_)
	# The root mean squared error
	print("Root Mean Squared Error: %.6f"
	      % math.sqrt(np.mean((rid.predict(data_X_test) - data_y_test) ** 2)))
	# Explained variance score: 1 is perfect prediction
	print("Variance score: %.6f" % rid.score(data_X_test, data_y_test))

	# Start plot
	plt.scatter(np.arange(56).reshape(1,56), rid.predict(data_X_test), 
    	color=colors[count], linewidth=1, label="alpha %.3f" % alpha)
	plt.plot(np.arange(56).reshape(56,1), abs(rid.predict(data_X_test)-data_y_test), 
		color=colors[count], linewidth=1)

# Plot the actual points
plt.scatter(np.arange(train_set_vol).reshape(1,train_set_vol), data_y_test,  
	color='black', label='Actual Points', linewidth=2, marker = 'x')

# Add up some explanations to the figure
plt.title('Ridge regression with different $\\alpha$')
plt.xlabel('$n^{th}$ point')
plt.ylabel('Predicted Point & Residual')
plt.legend(loc='upper left')
plt.xticks([0,14,28,42,56])
plt.yticks([0, 3, 6, 9, 12, 15, 30, 45])

plt.show()
