import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from numpy import genfromtxt
from sklearn import cross_validation, linear_model
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

# get the best degree of freedom
degrees = range(1,10)
rmse = []
print "Testing for the best degree..."

for deg in range(len(degrees)):
    polyFeatures = PolynomialFeatures(degree= degrees[deg],interaction_only=True, include_bias= False)
    lm = linear_model.LinearRegression()

    pipeline = Pipeline([("polynomial_features", polyFeatures),
                             ("linear_regression", lm)])

    pipeline.fit(my_feature,my_target)
    scores = cross_validation.cross_val_score(pipeline, my_feature, my_target, cv=10)

    rmse.append(np.mean(abs(scores)**0.5))

# plot graph for RMSE vs. degree of polynomial
plt.figure(1)
plt.plot(degrees,rmse)
plt.ylabel("RMSE")
plt.xlabel("Polynomial Degree")
plt.title("RMSE vs Polynomial Degree")


degrees = [degrees[rmse.index(min(rmse))]]
rmse = []

# apply transformation on dataset to the best degree of freedom and then find error
for deg in range(len(degrees)):
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(my_feature, my_target,  random_state=random.randrange(0, 100), test_size=0.1)
    polyFeatures = PolynomialFeatures(degree= degrees[deg],interaction_only=True, include_bias= False)
    X_train_trans = polyFeatures.fit_transform(data_X_train)
    X_test_trans = polyFeatures.fit_transform(data_X_test)

    lm = linear_model.LinearRegression()
    lm.fit(X_train_trans,data_y_train)
    predicted = lm.predict(X_test_trans)

    rmse.append(np.mean(abs(predicted-data_y_test)**0.5))

plt.figure(1)
plt.scatter(degrees,rmse)
plt.ylabel("RMSE")
plt.xlabel("Polynomial Degree")
plt.title("RMSE vs Polynomial Degree")

print "Best Degree of Polynomial : " + str(degrees[0])
print "Root Mean Squared Error for Best Parameters : " + str(rmse[0])

plt.show()
