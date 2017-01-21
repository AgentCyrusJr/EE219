# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from numpy import genfromtxt
import math

# Load the diabetes dataset
my_data = genfromtxt('housing_data.csv', delimiter=',')
print my_data.shape

# Split the data into training/testing sets
data_X_train = my_data[:450,:13]
data_X_test = my_data[450:,:13]

# Split the targets into training/testing sets
data_y_train = my_data[:450,13]
data_y_test = my_data[450:,13]

print data_y_train, data_y_test
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_y_train)

# The coefficients
print('Coefficients:', regr.coef_)
# The mean squared error
print("Root Mean Squared Error: %.2f"
      % math.sqrt(np.mean((regr.predict(data_X_test) - data_y_test) ** 2)))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % regr.score(data_X_test, data_y_test))

# Plot outputs
plt.title('Original Least Square')
plt1 = plt.figure(1)
plt2 = plt.figure(2)
plt3 = plt.figure(3)
plt1.scatter(np.arange(56).reshape(1,56), data_y_test,  color='black')
plt2.scatter(np.arange(56).reshape(1,56), regr.predict(data_X_test),  color='red')
plt3.plot(np.arange(56).reshape(56,1), regr.predict(data_X_test)-data_y_test, color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
