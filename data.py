import matplotlib.pyplot as plt
import datetime
import pandas as pd

data = pd.read_csv('http://localhost:5000/api/rest/currencies/bitcoin.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
ax1.set_ylabel('Closing Price ($)', fontsize=12)
ax2.set_ylabel('Volume ($ bn)', fontsize=12)
ax2.set_yticks([int('%d000000000' % i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 7]])
ax1.plot(data['Date'].astype(datetime.datetime), data['Open'])
ax2.bar(data['Date'].astype(datetime.datetime).values, data['Volume'].values)
fig.tight_layout()
plt.show()
