import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math, time
from math import sqrt
import os
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import tensorflow as tf

src_dataset = pd.read_csv('test_data.csv')
src_dataset['Date'] = src_dataset.index
plt.plot(src_dataset['Open'])
plt.savefig('as.png')
src_dataset['Date'] = pd.to_datetime(src_dataset['Date'])

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
dataset = min_max_scaler.fit_transform(src_dataset['Close/Last'].values.reshape(-1, 1))

train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

x_train, y_train = create_dataset(train, look_back=15)
x_test, y_test = create_dataset(test, look_back=15)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

look_back = 15
epochs = 100
batch_size = 1
units = 10000
model = Sequential()
model.add(LSTM(units=units,return_sequences=True, input_shape=(1, look_back)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size= batch_size, verbose=2)

trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
# invert predictions
trainPredict = min_max_scaler.inverse_transform(trainPredict.reshape(-1,1))
trainY = min_max_scaler.inverse_transform([y_train])
testPredict = min_max_scaler.inverse_transform(testPredict.reshape(-1,1))
testY = min_max_scaler.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(min_max_scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.text(-0.1, 1.1, '10 LSTM layer with 100 units / 10 dropout layer 0.2/ 1 dense layer, epochs={}, batch_size={}, lookback={}'.format(epochs, batch_size, look_back), transform=plt.gca().transAxes, fontsize=8, color='black')
plt.text(0.05, 0.9, f'Trainscore: {trainScore:.4f} RMSE', transform=plt.gca().transAxes, fontsize=12, color='black')
plt.text(0.05, 0.85, f'Testscore: {testScore:.4f} RMSE', transform=plt.gca().transAxes, fontsize=12, color='black')
output_file_name = f'prediction_plot_epochs{epochs}_batch{batch_size}_lookback{look_back}.png'
plt.savefig(output_file_name, dpi=300)


