import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


def build_model(inputs, output_size, neurons, activ_func="linear", dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


# bitcoin
btc_url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=" + time.strftime("%Y%m%d")
btc_market_info = pd.read_html(btc_url)[0]
btc_market_info = btc_market_info.assign(Date=pd.to_datetime(btc_market_info['Date']))
btc_market_info.loc[btc_market_info['Volume'] == "-", 'Volume'] = 0
btc_market_info['Volume'] = btc_market_info['Volume'].astype('int64')
btc_market_info.columns = [btc_market_info.columns[0]] + ['bt_' + i for i in btc_market_info.columns[1:]]

# ethereum
eth_url = "https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end=" + time.strftime("%Y%m%d")
eth_market_info = pd.read_html(eth_url)[0]
eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
eth_market_info.columns = [eth_market_info.columns[0]] + ['eth_' + i for i in eth_market_info.columns[1:]]

# merge
market_info = pd.merge(btc_market_info, eth_market_info, on=['Date'])
market_info = market_info[market_info['Date'] >= '2016-01-01']
for coins in ['bt_', 'eth_']:
    kwargs = {coins + 'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
    market_info = market_info.assign(**kwargs)
market_info.head()

split_date = '2018-01-01'

# prepare
for coins in ['bt_', 'eth_']:
    kwargs = {coins + 'close_off_high': lambda x: 2 * (x[coins + 'High'] - x[coins + 'Close']) / (
            x[coins + 'High'] - x[coins + 'Low']) - 1,
              coins + 'volatility': lambda x: (x[coins + 'High'] - x[coins + 'Low']) / (x[coins + 'Open'])}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['Date'] + [coin + metric for coin in ['bt_', 'eth_']
                                     for metric in ['Close', 'Volume', 'close_off_high', 'volatility']]]
model_data = model_data.sort_values(by='Date')

training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

window_len = 10
norm_cols = [coin + metric for coin in ['bt_', 'eth_'] for metric in ['Close', 'Volume']]

LSTM_training_inputs = []
for i in range(len(training_set) - window_len):
    temp_set = training_set[i:(i + window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['eth_Close'][window_len:].values / training_set['eth_Close'][
                                                                         :-window_len].values) - 1

LSTM_test_inputs = []
for i in range(len(test_set) - window_len):
    temp_set = test_set[i:(i + window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['eth_Close'][window_len:].values / test_set['eth_Close'][:-window_len].values) - 1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

np.random.seed(202)
# initialise model architecture
bt_model = build_model(LSTM_training_inputs, output_size=1, neurons=20)
# train model on data
# note: eth_history contains information on the training error per epoch
bt_history = bt_model.fit(LSTM_training_inputs,
                          (training_set['bt_Close'][window_len:].values / training_set['bt_Close'][
                                                                          :-window_len].values) - 1,
                          epochs=50, batch_size=1, verbose=2, shuffle=True)
bt_model.save('model.h5')

# run
fig, ax1 = plt.subplots(1, 1)
ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
ax1.set_xticklabels([datetime.date(2017, i + 1, 1).strftime('%b %d %Y') for i in range(12)])
ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][10:].astype(datetime.datetime),
         test_set['bt_Close'][window_len:], label='Actual')
ax1.plot(model_data[model_data['Date'] >= split_date]['Date'][10:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_test_inputs)) + 1) * test_set['bt_Close'].values[:-window_len])[0],
         label='Predicted')
ax1.annotate('MAE: %.4f' % np.mean(np.abs((np.transpose(bt_model.predict(LSTM_test_inputs)) + 1) - \
                                          (test_set['bt_Close'].values[window_len:]) / (
                                              test_set['bt_Close'].values[:-window_len]))),
             xy=(0.75, 0.9), xycoords='axes fraction',
             xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction', fontsize=13)
ax1.set_ylabel('Bitcoin Price ($)', fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.show()
