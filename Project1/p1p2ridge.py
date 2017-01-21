# Code source: Jaques Grobler
# License: BSD 3 clause

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from numpy import genfromtxt


# Load the diabetes dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')
print my_data.shape

# Split the data into training/testing sets
data_X_train = my_data[:450,:13]
data_X_test = my_data[450:,:13]

# Split the targets into training/testing sets
data_y_train = my_data[:450,13]
data_y_test = my_data[450:,13]

colors = ['teal', 'yellowgreen', 'gold', 'red']
#plot the test points
plt.scatter(np.arange(56).reshape(1,56), data_y_test,  color='black', label='actual')

# Create Ridge regression object
for count, alpha in enumerate([1, 0.1, 0.01, 0.001]):
	rid = linear_model.Ridge(alpha, fit_intercept = False)
	# Train the model using the training sets
	rid.fit(data_X_train, data_y_train)
	plt.scatter(np.arange(56).reshape(1,56), rid.predict(data_X_test), 
    	color=colors[count], linewidth=2, label="alpha %.3f" % alpha)
	# The coefficients
	print('Coefficients:', rid.coef_)
	# The mean squared error
	print("Root Mean Squared Error: %.6f"
	      % math.sqrt(np.mean((rid.predict(data_X_test) - data_y_test) ** 2)))
	# Explained variance score: 1 is perfect prediction
	print("Variance score: %.6f" % rid.score(data_X_test, data_y_test))

plt.legend(loc='lower left')
plt.xticks([0,14,28,42,56])
plt.yticks([10,20,30])

plt.show()
