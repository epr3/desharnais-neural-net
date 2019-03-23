from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.utils.vis_utils import plot_model
import csv
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

X_train = X_train.reshape(len(X_train), 1, 9)
X_test = X_test.reshape(len(X_test), 1, 9)

# TODO: test different layers
model=Sequential()
model.add(LSTM(50,input_shape=(1, 9), return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adamax',metrics=['accuracy', 'mae', 'mse', 'mape'])

print(model.summary())

model.fit(X_train,Y_train,validation_data=(X_test, Y_test),batch_size=20,epochs=10000,verbose=1)

model_json = model.to_json()
with open("lstm.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("lstm.h5")

Y_pred = model.predict(X_test)

print(sc.inverse_transform(Y_test))
print(sc.inverse_transform(Y_pred))

print(mean_squared_error(Y_test, Y_pred))
print(mean_absolute_error(Y_test, Y_pred))
print(r2_score(Y_test, Y_pred))

with open('lstm.csv', mode='w') as lstm_file:
    lstm_writer = csv.writer(lstm_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    D_Y_test = sc.inverse_transform(Y_test)
    D_Y_pred = sc.inverse_transform(Y_pred)

    lstm_writer.writerow(['Actual Effort', 'Predicted Effort', 'MRE'])
    for i in range(0, len(D_Y_test)):
      lstm_writer.writerow([D_Y_test[i], D_Y_pred[i], abs(D_Y_test[i] - D_Y_pred[i])/D_Y_test[i]])

plot_model(model, to_file='lstm_plot.png', show_shapes=True, show_layer_names=True)