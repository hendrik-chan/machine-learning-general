#https://machinelearningmastery.com/selectively-scale-numerical-input-variables-for-machine-learning/

# evaluate a logistic regression model on the raw diabetes dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values

#plot histogram to find the distribution of each inputs
print("plotting histogram")
print(dataframe.shape)
# histograms of the variables
#dataframe.hist()
#pyplot.show()

# separate into input and output elements
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))

print(y[1:5])

# define the model
model = LogisticRegression(solver='liblinear')
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
m_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Model #1 Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))



##Model #2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
# separate into input and output elements
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the modeling pipeline
model = LogisticRegression(solver='liblinear')
scaler = MinMaxScaler()
pipeline = Pipeline([('s',scaler),('m',model)])
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Model 2 Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))

##Model #3
from sklearn.preprocessing import StandardScaler
# separate into input and output elements
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the modeling pipeline
model = LogisticRegression(solver='liblinear')
scaler = StandardScaler()
pipeline = Pipeline([('s',scaler),('m',model)])
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Model 3 Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))

##Model 4 selecting normalization no non-gaussian inputs
from sklearn.compose import ColumnTransformer
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# define column indexes for the variables with "normal" and "exponential" distributions
norm_ix = [1, 2, 5]
exp_ix = [0, 3, 4, 6, 7]
# define the selective transforms
t = [('e', MinMaxScaler(), exp_ix)]
selective = ColumnTransformer(transformers=t, remainder='passthrough')
# define the modeling pipeline
model = LogisticRegression(solver='liblinear')
pipeline = Pipeline([('s',selective),('m',model)])
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Model 4 Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))


#Model 5 selecting standardization no non-gaussian inputs
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# define column indexes for the variables with "normal" and "exponential" distributions
norm_ix = [1, 2, 5]
exp_ix = [0, 3, 4, 6, 7]
# define the selective transforms
t = [('e', StandardScaler(), exp_ix)]
selective = ColumnTransformer(transformers=t, remainder='passthrough')
# define the modeling pipeline
model = LogisticRegression(solver='liblinear')
pipeline = Pipeline([('s',selective),('m',model)])
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Model 5 Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))


#Model 6 selective normalization and standardization
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# define column indexes for the variables with "normal" and "exponential" distributions
norm_ix = [1, 2, 5]
exp_ix = [0, 3, 4, 6, 7]
# define the selective transforms
t = [('e', MinMaxScaler(), exp_ix), ('n', StandardScaler(), exp_ix) ]
selective = ColumnTransformer(transformers=t, remainder='passthrough')
# define the modeling pipeline
model = LogisticRegression(solver='liblinear')
pipeline = Pipeline([('s',selective),('m',model)])
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Model 5 Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
