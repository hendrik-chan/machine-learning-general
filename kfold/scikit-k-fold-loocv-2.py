# https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
# loocv to manually evaluate the performance of 1 randomforest and 1 regression problem
# Note we also use xgboos
from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X = data[:, :-1]
y = data[:,-1]
print(X.shape, y.shape)

# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestClassifier(random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('cross_val_score Random Forest Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#Using XGBoost
xgmodel = XGBClassifier()
scores = cross_val_score(xgmodel, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('cross_val_score xgboost Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


#housing data regression
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
# split into inputs and outputs
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# create loocv procedure
cv = LeaveOneOut()
# create model
model = RandomForestRegressor(random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force positive
scores = absolute(scores)
# report performance
print('random forest regressor MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

#Using XGBoost Regressor
xgmodel = XGBRegressor()
scores = cross_val_score(xgmodel, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force positive
scores = absolute(scores)
# report performance
print('cross_val_score xgboostregressor MAE: %.3f (%.3f)' % (mean(scores), std(scores)))