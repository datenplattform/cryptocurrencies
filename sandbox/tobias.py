import pandas as pd
import numpy as np


def get_training_data():
    training_data = pd.read_csv("../input/bitcoin_price_Training - Training.csv", index_col='Date')
    training_data.index = pd.to_datetime(training_data.index)
    training_data = training_data.sort_index()
    training_data.reset_index(drop=True)
    training_data = training_data['Close'].values.astype('float32')

    return training_data


def get_test_data():
    test_data = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv", index_col='Date')
    test_data.index = pd.to_datetime(test_data.index)
    test_data = test_data.sort_index()
    test_data.reset_index(drop=True)
    test_data = test_data['Close'].values.astype('float32')

    return test_data


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay]
        yield samples, targets
