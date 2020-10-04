#https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

from pandas import read_csv
from matplotlib import pyplot
series = read_csv('sunspots.csv', header=0, index_col=0)
print(series.head())
#series.plot()
#pyplot.show()

#SImple train test split. Note the difference with typical train_test_split
# is that there is a split point here.
# The split point can be calculated as a specific index in the array. 
#All records up to the split point are taken as the training dataset and 
#all records from the split point to the end of the list of observations are taken as the test set.
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
#pyplot.plot(train)
#pyplot.plot([None for i in train] + [x for x in test])
#pyplot.show()

#Multiple train test splits
index = 1
for train_index, test_index in splits.split(X):
	train = X[train_index]
	test = X[test_index]
	print('Observations: %d' % (len(train) + len(test)))
	print('Training Observations: %d' % (len(train)))
	print('Testing Observations: %d' % (len(test)))
	pyplot.subplot(310 + index)
	pyplot.plot(train)
	pyplot.plot([None for i in train] + [x for x in test])
	index += 1
pyplot.show()

#Walk forward testing
n_train = 500
n_records = len(X)
for i in range(n_train, n_records):
	train, test = X[0:i], X[i:i+1]
	print('train=%d, test=%d' % (len(train), len(test)))