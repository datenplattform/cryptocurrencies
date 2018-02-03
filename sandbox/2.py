from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import seaborn as sns
import quandl

data = quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
print(data)
pyplot.plot(data['Date'], data['Open'])
pyplot.show()
