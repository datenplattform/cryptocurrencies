# Load libs
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.utils import plot_model

btc = pd.read_csv('cryptocurrencypricehistory/bitcoin_price.csv')
btc = btc.iloc[::-1]

eth = pd.read_csv('cryptocurrencypricehistory/ethereum_price.csv')
eth = eth.iloc[::-1]

features = btc[["Open", "High", "Low", "Close"]].values

# we change the data to have something more generalizeable, lets say [ %variation , %high, %low]
price_variation = (1 - (features[:, 0] / features[:, 3])) * 100
highs = (features[:, 1] / np.maximum(features[:, 0], features[:, 3]) - 1) * 100
lows = (features[:, 2] / np.minimum(features[:, 0], features[:, 3]) - 1) * 100

X_train = np.array([price_variation, highs, lows]).transpose()
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

Y_train = np.array((np.sign((features[2:, 3] / features[:-2, 3] - 1)) + 1) / 2)

model = Sequential()
model.add(LSTM(100, input_shape=(None, 1), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(50))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="mse", optimizer="rmsprop")
plot_model(model, to_file='model.png', show_shapes=True)

model.fit(X_train[:-2], Y_train, batch_size=512, epochs=500, validation_split=0.05)

# we change the data to have something more generalizeable, lets say [ %variation , %high, %low]
price_variation = (1 - (features[:, 0] / features[:, 3])) * 100
highs = (features[:, 1] / np.maximum(features[:, 0], features[:, 3]) - 1) * 100
lows = (features[:, 2] / np.minimum(features[:, 0], features[:, 3]) - 1) * 100

X_test = np.array([price_variation, highs, lows]).transpose()
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.array((np.sign(features[2:, 3] / features[:-2, 3] - 1) + 1) / 2)
print(Y_test[:3])

model.evaluate(X_test[:-2], Y_test)

pred = model.predict(X_test)

predicted = (np.sign(pred - 0.45) + 1) / 2 * 50

start = 650
stop = 700
plt.plot(predicted[start:stop], 'r')  # prediction is in red.
plt.plot(features[start:stop, 3], 'b')  # actual in blue.
plt.plot(Y_test[start:stop] * 50)
plt.show()

model.save('model.h5')