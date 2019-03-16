from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()

import pandas as pd
import numpy as np

# Read data
data = pd.read_csv('./desharnais.csv')
Y = data.iloc[:, 6].values
data = data.drop('YearEnd', axis=1)
data = data.drop('Effort', axis=1)
X = data.iloc[:,2:]
Y = Y.reshape(-1, 1)

X_normalised = sc.fit_transform(X)
Y_normalised = sc.fit_transform(Y)

total_length = len(data)
train_length = int(0.8*total_length)
test_length = int(0.2*total_length)

X_train = X_normalised[:train_length]
X_test = X_normalised[train_length:]
Y_train = Y_normalised[:train_length]
Y_test = Y_normalised[train_length:]

# TODO: test different layers
model=Sequential()
model.add(Dense(9,input_dim=9,activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy', 'mae'])

model.summary()

model.fit(X_train,Y_train,validation_data=(X_test, Y_test),batch_size=20,epochs=10,verbose=1)

Y_pred = model.predict(X_test)

print(sc.inverse_transform(Y_test))
print(sc.inverse_transform(Y_pred))