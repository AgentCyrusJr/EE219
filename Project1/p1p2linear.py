import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from numpy import genfromtxt
import math

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

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_y_train)

# The coefficients
print('Coefficients:', regr.coef_)
# The root mean squared error
print("Root Mean Squared Error: %.3f"
      % math.sqrt(np.mean((regr.predict(data_X_test) - data_y_test) ** 2)))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.3f" % regr.score(data_X_test, data_y_test))

# Add up some explanations to the figure
plt.title('Ordinary Least Square')
plt.xlabel('$n^{th}$ point')
plt.ylabel('Predicted Point & Residual')
plt.scatter(np.arange(56).reshape(1,56), data_y_test,  color='black', label='Actual Point', linewidth=2, marker = 'x')
plt.scatter(np.arange(56).reshape(1,56), regr.predict(data_X_test),  color='red', label='Predict Point', linewidth=1)
plt.plot(np.arange(56).reshape(56,1), abs(regr.predict(data_X_test)-data_y_test), color='blue',
         linewidth=1, label='residual')

plt.legend(loc='upper left')
plt.xticks([0,14,28,42,56])
plt.yticks([0,2,4,6,8,10,20,30])

plt.show()
