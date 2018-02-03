from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.utils import to_categorical
import numpy as np

x_train = np.random.random((1000, 20))
y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

csv_logger = CSVLogger('training.log')
model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[csv_logger])
score = model.evaluate(x_test, y_test, batch_size=128)

print(model.summary())
print(model.get_config())
print(model.get_weights())
print(model.to_json())
