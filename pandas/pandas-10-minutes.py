# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

### Pandas object creation
import pandas as pd
import numpy as np

#create pd series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

dates = pd.date_range('20130101', periods=6)
#Create from numpy object, the columns of the df will be A,B,C,D. Dates is for what?
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

#Create dataframe from Dictionary, note pandas allow each column to have different dtypes
df2 = pd.DataFrame({'A': 1.,
   'B': pd.Timestamp('20130102'),
   'C': pd.Series(1, index=list(range(4)), dtype='float32'),
   'D': np.array([3] * 4, dtype='int32'),
   'E': pd.Categorical(["test", "train", "test", "train"]),
'F': 'foo'})
print(df2.dtypes)

####Viewing data
df.head(5)
df.tail(5)
df.index
df.columns

#Note that this can be an expensive operation when your DataFrame has columns with different data types, 
#which comes down to a fundamental difference between pandas and NumPy: NumPy arrays have one dtype for the entire array, while pandas DataFrames have one dtype per column
df.to_numpy() #df.values also changes to numpy array but to_numpy is recommended now, so must always use this to pass to ML Model

df.describe() # show min, max, mean per column , etc
df.T # transpose

df.sort_index(axis=1, ascending=False) #Sorting by index
df.sort_values(by='B') #sorting by column B


#####Getting data
df['A'] #retrieving one column

df[0:3] #Slicing by row , first 3 rows
df['20130102':'20130104'] #Slicing by row index , note because index is date.

#selection by label
df.loc[:, ['A', 'B']] #all rows, but only column A and B
df.loc['20130102':'20130104', ['A', 'B']]

#Selection by position (integer based)
df.iloc[3] #Row 3
df.iloc[3:5, 0:2] 
df.iloc[[1, 2, 4], [0, 2]]

#Boolean indexing
df[df['A'] > 0] #Select WHERE A > 0

#df2[df2['E'].isin(['two', 'four'])] #Using is IN, is like select where IS IN a list

#Update or setting value, like setting value of the table.
df.at[dates[0], 'A'] = 0
df2 = df.copy()
df2[df2 > 0] = -df2 #Update where df2 > 0

#Handling missing data (PANDAS)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E']) #This basically just does copying.
df1.loc[dates[0]:dates[1], 'E'] = 1

df1.dropna(how='any') #Drop any row that has missing values
df1.fillna(value=5) #Filling missing data

#Select any rows that have specific column blank.
#https://stackoverflow.com/questions/40245507/python-pandas-selecting-rows-whose-column-value-is-null-none-nan
#empty_email = dataset[dataset['email'].isna()]
#empty_pbpass = dataset[dataset['s_pbpass_type'].isna()]