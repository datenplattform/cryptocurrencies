import pandas as pd
import numpy as np
from pylab import plt

data_url = 'http://www.quandl.com/api/v1/datasets/BCHARTS/BITSTAMPUSD.csv'
bitcoin = pd.read_csv(data_url, index_col='Date').iloc[::-1]
bitcoin['Close'].plot(figsize=(12, 8), grid=True, title="BTC/USD")

plt.style.use('ggplot')


class SMAVectorBacktester(object):
    def __init__(self, name, start, end, SMA1, SMA2, tc):
        self.name = name
        self.start = start
        self.end = end
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.tc = tc
        self.get_data()

    def get_data(self):
        raw = pd.read_csv(self.name + '.csv', index_col=0).iloc[::-1]
        raw = raw[(raw.index >= self.start) & (raw.index < self.end)]
        raw['SMA1'] = raw['Value'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['Value'].rolling(self.SMA2).mean()
        raw['Returns'] = np.log(raw['Value'] / raw['Value'].shift(1))
        self.data = raw.dropna()

    def plot_data(self):
        self.data[['Value', 'SMA1', 'SMA2']].plot(figsize=(10, 6), title=self.name)

    def run_strategy(self):
        data = self.data.copy()
        data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['Strategy'] = data['Position'].shift(1) * data['Returns']
        data.dropna(inplace=True)
        trades = (data['Position'].shift(1) * data['Returns'])
        data['Strategy'] = np.where(trades, data['Strategy'] - self.tc, data['Strategy'])
        data['CReturns'] = data['Returns'].cumsum().apply(np.exp)
        data['CStrategy'] = data['Strategy'].cumsum().apply(np.exp)
        self.results = data
        return data[['CReturns', 'CStrategy']].ix[-1]

    def plot_results(self):
        self.results[['CReturns', 'CStrategy']].plot(figsize=(10, 6))


sma = SMAVectorBacktester(name='bitcoin', start='2012-01-01', end='2017-07-08', SMA1=42, SMA2=252, tc=0.00001)
sma.plot_data()
sma.run_strategy()
sma.plot_results()

plt.show()
