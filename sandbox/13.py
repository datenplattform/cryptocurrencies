import pandas as pd

items = pd.read_html('https://coinmarketcap.com/gainers-losers/', index_col=None)


print(items)