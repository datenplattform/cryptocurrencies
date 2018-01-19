from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import plot_model

model = Sequential()
model.add(LSTM(20, input_shape=(10, 64)))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation("linear"))

model.compile(loss="mae", optimizer="adam")
plot_model(model, to_file='model.png', show_shapes=True)
