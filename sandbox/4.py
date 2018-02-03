import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pylab as plt
from tobias import get_training_data, get_test_data, generator
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

training_data = get_training_data()
test_data = get_test_data()

train = training_data.reshape(-1, 1)
test = test_data.reshape(-1, 1)


print(training_data)
print(train)