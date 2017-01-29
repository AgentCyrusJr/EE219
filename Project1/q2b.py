import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# q2.b

df = pd.read_csv('network_backup_dataset.csv', ',')
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

# initial values
best_num_of_trees = 20
best_max_depth = 4
best_max_features = 6

print "################### Compute for best max_depth#################################"
param_grid2 = {'max_depth': range(4, 15, 1)}
random_forest = RandomForestRegressor(n_estimators=best_num_of_trees, max_features=best_max_features)
gsearch2 = GridSearchCV(estimator=random_forest, param_grid=param_grid2, scoring='neg_mean_squared_error', cv=10)
gsearch2.fit(X, y)
results2 = gsearch2.cv_results_
print results2['mean_test_score']
RMSE_scores2 = np.sqrt(-1 * results2['mean_test_score'])
best_RMSE = np.sqrt(-1 * gsearch2.best_score_)
max_depth_array = param_grid2['max_depth']
best_max_depth = gsearch2.best_params_['max_depth']
print "RMSE_scores", RMSE_scores2
print "best_RMSE", best_RMSE
print "max_depth_array", max_depth_array
print "best_max_depth", best_max_depth
plt.title("RMSE - max_depth")
plt.plot(max_depth_array, RMSE_scores2)
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.savefig('RMSE - max_depth.png')
plt.show()

print "################### Compute for best num_of_tree(n_estimators)##################"
param_grid1 = {'n_estimators': range(20, 151, 10)}
random_forest = RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features)
gsearch1 = GridSearchCV(estimator=random_forest,
                        param_grid=param_grid1, scoring='neg_mean_squared_error', cv=10)
gsearch1.fit(X, y)
results1 = gsearch1.cv_results_
print results1['mean_test_score']
RMSE_scores1 = np.sqrt(-1 * results1['mean_test_score'])
num_of_trees_array = param_grid1['n_estimators']
best_num_of_trees = gsearch1.best_params_['n_estimators']
best_RMSE = np.sqrt(-1 * gsearch1.best_score_)
print "RMSE_scores", RMSE_scores1
print "best_RMSE", best_RMSE
print "num_of_trees_array", num_of_trees_array
print "best_num_of_trees", best_num_of_trees
plt.title("RMSE - num_of_trees")
plt.plot(num_of_trees_array, RMSE_scores1)
plt.xlabel('num_of_trees')
plt.ylabel('RMSE')
plt.savefig('RMSE - num_of_trees.png')
plt.show()

print "################### Compute for best max_features#################################"
param_grid3 = {'max_features': range(6, 0, -1)}
random_forest = RandomForestRegressor(n_estimators=best_num_of_trees, max_depth=best_max_depth)
gsearch3 = GridSearchCV(estimator=random_forest, param_grid=param_grid3, scoring='neg_mean_squared_error', cv=10)
gsearch3.fit(X, y)
results3 = gsearch3.cv_results_
print results3['mean_test_score']
RMSE_scores3 = np.sqrt(-1 * results3['mean_test_score'])
best_RMSE = np.sqrt(-1 * gsearch3.best_score_)
max_features_array = param_grid3['max_features']
best_max_features = gsearch3.best_params_['max_features']
print "RMSE_scores", RMSE_scores3
print "best_RMSE", best_RMSE
print "max_features_array", max_features_array
print "best_max_features", best_max_features
plt.title("RMSE - max_features")
plt.plot(max_features_array, RMSE_scores3)
plt.xlabel('max_features')
plt.ylabel('RMSE')
plt.savefig('RMSE - max_features.png')
plt.show()

print "################### Tuning Result #################################"
print "best_num_of_trees:", best_num_of_trees
print "best_max_depth:", best_max_depth
print "best_max_features", best_max_features
print "best_RMSE", best_RMSE

print "################### Recompute with tunned parameters for coefficients#################################"
random_forest = RandomForestRegressor(n_estimators=best_num_of_trees, max_depth=best_max_depth,
                                      max_features=best_max_features)
random_forest.fit(X, y)
# feature_importances_ : array of shape = [n_features]
# The feature importances (the higher, the more important the feature).
print random_forest.feature_importances_
scores = cross_val_score(random_forest, X, y, cv=10, scoring='neg_mean_squared_error')
print "Final RMSE: ", np.sqrt(-1 * np.mean(scores))
scores = cross_val_score(random_forest, X, y, cv=10)
print "R2 score: ", np.mean(scores)
