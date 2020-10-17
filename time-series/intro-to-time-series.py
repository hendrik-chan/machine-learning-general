#https://www.kdnuggets.com/2020/09/introduction-time-series-analysis-python.html
import pandas as pd
import numpy as np

#Note: setting the date column as index
df = pd.read_csv('./UMTMVS.csv', index_col='DATE')
print(df.head())
print(df.index)

###Convert the index from object to date type
df.index = pd.to_datetime(df.index)
print(df.index)

##Or technically can just use this line: df = pd.read_csv(‘Data/UMTMVS.csv’, index_col=’DATE’, parse_dates=True)

##Selecting by period
df.loc['2000-01-01':'2015-01-01']

#Select data by every 12 months
df.loc['1992-01-01':'2000-01-01':12]


##TIme-series resampling
print(df.resample(rule='AS').mean().head())
print(df.loc['1992-01-01'].count())

##Time-series rolling window
#the rolling average of 10 days
print(df.rolling(window=10).mean().head(20)) # head to see first 20 values, note that first 9 values will show NaN because it doesnt have Mean.

#The maximum value from a window of 30 days
print(df.rolling(window=30).max()[30:].head(20)) # head is just to check top 20 values, [30:] is to get the 30th entry onwards.


