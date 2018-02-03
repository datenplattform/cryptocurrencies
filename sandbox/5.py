from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

train_df = pd.read_csv(r'../input/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv(r'../input/fashionmnist/fashion-mnist_test.csv')

train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=12345)

image = x_train[50, :].reshape((28, 28))

# CNN
image_rows = 28
image_cols = 28
batch_size = 512
image_shape = (image_rows, image_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *image_shape)
x_test = x_test.reshape(x_test.shape[0], *image_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *image_shape)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

tensorboard = TensorBoard(log_dir=r'../logs/5', write_graph=True, write_grads=True, write_images=True, histogram_freq=1)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=2, validation_data=(x_validate, y_validate),
          callbacks=[tensorboard])

model.evaluate(x_test, y_test, verbose=0)
