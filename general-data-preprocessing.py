#https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html

#1 train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#2 taking care of missing values - this will return list of columns and sum of missing values
df.isna().sum()

#Impute missing values using Mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(fill_value=np.nan, startegy='mean')
X = imputer.fit_transform(df)
#but because it returns ndarray, we need to convert back to dataframe.
X = pd.DataFrame(X, columns=df.columns)

#Completely drop the missing rows
droppedDF = df.dropna()

#3 taking care of categorical features

#Using labelencoder, will convert to 0,1,2
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
l1.fit(catDf['Country'])
catDf.Country = l1.transform(catDf.Country) # Note: we save it back to catDF

#Using oneHotEncoder, there are two ways,first we just use panda.get_dummies 
#or second way we use scikit learn onehotencoding
catDf = pd.get_dummies(data=catDf)

from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
s1 = pd.dataFrame(oh.fit_transform(catDf.iloc[:, [0,3]])) #We onehotencode column 0 and 3 which both are categorical
    #fit_transform returns ndarray, thats why we convert it to pd dataframe again
catDf = pd.concat([catDf,s1], axis=1) #we add it back? but we still have the original columns though

#4 normalizing and standardizing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
catDf.iloc[:,1:-1] = ss.fit_transform(catDf.iloc[:,1:-1]) #only standardize column 2 and 3 (excluding last column)
print(catDf)

from sklearn.preprocessing import Normalizer
norm = Normalizer()
catDf.iloc[:,1:-1] = norm.fit_transform(catDf.iloc[:,1:-1])


