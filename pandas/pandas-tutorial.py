# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:38:28 2020

@author: hTjang
"""

#https://www.youtube.com/watch?v=xvpNA7bC8cs
import pandas as pd
ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.head(3)

#Using loc for filtering rows and selecting columns by label
ufo.loc[0 , :] # i want row 0, all columns, : means ALL columns

ufo.loc[[0,1,2], :] # i want row 0,1,2

#remember loc : is inclusion on both sides, 0,1,2
ufo.loc[0:2,:] # i want row 0 through 2 : 0,1,2 (continous selection)
ufo.loc[0:2] #same as above

ufo.loc[:, ['City', 'State']] #i want all rows and column City,State
ufo.loc[:, 'City':'State'] #i want all rows, column city through State

#you can achieve the same thing with head and drop
ufo.head(3).drop('Time', axis=1)

#using loc with boolean condition
ufo[ufo.City =='Oakland'] 
ufo[ufo.City =='Oakland'].State 
ufo.loc[ufo.City =='Oakland', :] # you can do the same thing with loc
ufo.loc[ufo.City =='Oakland', 'State']

#the difference is that ufo[ufo.City =='Oakland'].State needs 2 operations
#ufo.loc[ufo.City =='Oakland', 'State'] is only 1 operation

############# iloc for filtering rows and selecting columns by Integer position
##Thats what i is for in iloc.
###??but loc also can use integers, no??
ufo.iloc[:,[0, 3]] #all rows, column 0 and 3
ufo.iloc[:,0:4] #all rows , columns 0 to 3.

####with Iloc its inclusive of 0 exclusive of 4, so its 0 to 3.
####It's the same with range(0,4) 
####Note the difference with .loc

ufo.iloc[0:3, :] #row 0,1,2

#Not recommended selections
ufo[['City','State']] # vs ufo.loc[:, ['City','State']]
ufo[0:2] #rows 0 to 1


############# Using ix allows you to select labels and integers
###its a blend between loc and iloc
###But note that .ix is deprecated
###.ix is deprecated. Please use
###.loc for label based indexing or
###.iloc for positional indexing
drinks = pd.read_csv('http://bit.ly/drinksbycountry', index_col = 'country')
drinks.head()

drinks.ix['Albania', 0] #select rows whose label is 'Albania'
drinks.ix[1, 'beer_servings'] #select row 1, and column name = beer_servings
###So when we use 'Albania' its the label / value of the row
###When we use 'beer_servings' its the column name.






