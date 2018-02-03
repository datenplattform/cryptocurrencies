import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

rcParams['figure.figsize'] = 15, 6
seaborn.set()

data = pd.read_csv("../input/bitcoin_price_Training - Training.csv", index_col='Date')
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# data['Close'].plot()
# data.resample('W').sum().plot()
# data.groupby(data.index.year).mean().plot()
# data.groupby(data.index.dayofweek).sum().plot()
# data.groupby(data.index.dayofweek).mean().plot()
# data.groupby(data.index.dayofyear).mean().plot()
# data.groupby(data.index.month).mean().plot()
# data.groupby(data.index.quarter).mean().plot()
data.groupby(data.index.quarter).plot()
data.groupby(data.index.quarter).mean().plot()

plt.show()
