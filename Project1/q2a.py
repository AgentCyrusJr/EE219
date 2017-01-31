import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
import statsmodels.api as sm

# q2.a
df = pd.read_csv('network_backup_dataset.csv', ',')
# preprocess data
df2 = df.replace({'Day of Week': {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5,
                                  'Saturday': 6, 'Sunday': 7},
                  'Work-Flow-ID': {'work_flow_0': 0, 'work_flow_1': 1, 'work_flow_2': 2, 'work_flow_3': 3,
                                   'work_flow_4': 4},
                  'File Name': {'File_0': 0, 'File_1': 1, 'File_2': 2, 'File_3': 3, 'File_4': 4, 'File_5': 5,
                                'File_6': 6, 'File_7': 7, 'File_8': 8, 'File_9': 9, 'File_10': 10, 'File_11': 11,
                                'File_12': 12, 'File_13': 13, 'File_14': 14, 'File_15': 15, 'File_16': 16,
                                'File_17': 17, 'File_18': 18, 'File_19': 19, 'File_20': 20, 'File_21': 21,
                                'File_22': 22,
                                'File_23': 23, 'File_24': 24, 'File_25': 25, 'File_26': 26, 'File_27': 27,
                                'File_28': 28, 'File_29': 29, }})
X = df2[
    ['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 'Work-Flow-ID', 'File Name', 'Backup Time (hour)']]
y = df2['Size of Backup (GB)']

# use OLS model to compute p-value with other attributes.
model = sm.OLS(y, X)
results = model.fit()
print results.params
print(results.summary())

# regression without cross validation for the coefficients
# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
test_shape = y_test.shape[0]
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print "Coefficients:", linear_regression.coef_
print "Intercept:", linear_regression.intercept_
y_predict = linear_regression.predict(X_test)
print "RMSE without cross validation:", np.sqrt(metrics.mean_squared_error(y_test, y_predict))

# 10 fold cross validation, can try KFold if wanna iterative visit each train and test data
predicted = cross_val_predict(linear_regression, X, y, cv=10)
print "RMSE with cross validation:", np.sqrt(metrics.mean_squared_error(y, predicted))

# compute for scores
scores = cross_val_score(linear_regression, X, y, cv=10, scoring='neg_mean_squared_error')
print "RMSE with cross validation (computed by cross_val_score):", np.sqrt(-1 * np.mean(scores))
r2_scores = cross_val_score(linear_regression, X, y, cv=10, scoring='r2')
print "r2:", r2_scores.mean()

# plot fitted values vs actual values scattered plot
plt.scatter(y, predicted)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual values')
plt.ylabel('Fitted values')
plt.savefig('Fitted values vs Actual values.png')
plt.show()

# Plot Fitted values and actual values scattered plot over time
plt.title('Fitted values and actual values scattered plot over time')
plt.scatter(np.arange(test_shape), y_test, color='black', label='actual')
plt.scatter(np.arange(test_shape), y_predict, color='red', label='predict')
plt.xlabel('Time')
plt.ylabel('Fitted & Actual Values')
plt.legend(loc='upper left')
plt.savefig('Fitted values and actual values over time.png')
plt.show()

# plot residuals versus fitted values plot
plt.title('residuals versus fitted values plot')
plt.xlabel('Fitted values')
plt.ylabel('residuals')
plt.scatter(y_predict, y_predict - y_test, color='blue', lw=1, label='residual')
plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--', lw=4)
plt.legend(loc='upper left')
plt.savefig('residuals versus fitted values plot.png')
plt.show()
