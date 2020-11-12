#https://www.kdnuggets.com/2019/12/random-forest-vs-neural-networks-predicting-customer-churn.html

#Using random forest to predict churn

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import InputLayer 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.constraints import MaxNorm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(data.head())

data.SeniorCitizen.replace([0, 1], ["No", "Yes"], inplace= True)
data.TotalCharges.replace([" "], ["0"], inplace= True)
data.TotalCharges = data.TotalCharges.astype(float)
data.drop("customerID", axis= 1, inplace= True) 
data.Churn.replace(["Yes", "No"], [1, 0], inplace= True)

print(data.loc[1:3,:])

###Get_dummies will change from a Gender column to the following. Is it like one hot encoding??
#gender_Female	gender_Male	SeniorCitizen_No	SeniorCitizen_Yes
#1	0	1	0
#0	1	1	0
#0	1	1	0
#0	1	1	0

data = pd.get_dummies(data)
print(data.loc[1:3,:])
data.to_csv('out.csv',sep=',',)

X = data.drop("Churn", axis= 1)
y = data.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                              random_state=42)
rf.fit(X_train, y_train) 
score = rf.score(X_train, y_train)
score2 = rf.score(X_test, y_test)
print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))

y_pred = rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

##Print the feature importance by random forest
fi = pd.DataFrame({'feature': list(X_train.columns),
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()


###Next we compare with a NN, as its just tabular data, so we dont need too many layers to prevent overfitting.
model = Sequential()
model.add(Dense(64, input_dim=46, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(8, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(rate=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.title('model accuracy') 
plt.ylabel('accuracy')
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='upper left') 
plt.show()