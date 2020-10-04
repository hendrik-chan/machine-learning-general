#https://machinelearningmastery.com/binary-flags-for-missing-values-for-machine-learning/

#Compare 3 models
# baseline on horse colic with random forest classification. SimpleImpute with mean
# add a binary column to indicate whether a column had missing value or not
# add 1 binary column per input column

from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
# Note: Our Y is at column 23
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Model 1 Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#Model 2
from numpy import isnan
from numpy import hstack
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# sum each row where rows with a nan will sum to nan
a = X.sum(axis=1)
# mark all non-nan as 0
a[~isnan(a)] = 0
# mark all nan as 1
a[isnan(a)] = 1
a = a.reshape((len(a), 1))
# add to the dataset as another column
X = hstack((X, a))
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Model 2 Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


#model 3
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer(add_indicator=True)
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Model 3 Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))