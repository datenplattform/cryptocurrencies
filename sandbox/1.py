import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
from datetime import datetime

sns.set()


def convert(date):
    holder = []
    for i in date:
        holder.append(parse(i))

    return np.array(holder)


train = pd.read_csv("../input/bitcoin_price_Training - Training.csv")
train = train[::-1]
train['Date'] = convert(train['Date'].values)
train = train.set_index('Date')
train.isnull().any()

test = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv")
test = test[::-1]

print(train.tail())
