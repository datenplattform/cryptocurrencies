import pandas as pd
import time
from flask import Flask, jsonify, make_response

app = Flask(__name__)


def create_response(data, content_type):
    if content_type == 'csv':
        response = make_response(data.to_csv(index=False))
        response.headers["Content-type"] = "text/csv"

        return response
    elif content_type == 'json':
        response = make_response(data.to_json(orient='records'))
        response.headers["Content-type"] = "application/json"

        return response


def get_historical_data(currency):
    start = "20130428"
    end = time.strftime("%Y%m%d")
    url = "https://coinmarketcap.com/currencies/" + currency + "/historical-data/?start=" + start + "&end=" + end
    items = pd.read_html(url)

    df = items[0]
    df = df.assign(Date=pd.to_datetime(df['Date']))
    df.loc[df['Volume'] == "-", 'Volume'] = 0
    df['Volume'] = df['Volume'].astype('int64')
    df = df.assign(Difference=lambda x: (x['Close'] - x['Open']) / x['Open'])
    df = df.assign(Volatility=lambda x: (x['High'] - x['Low']) / (x['Open']))
    df = df.assign(ClosingHighPriceGap=lambda x: 2 * (x['High'] - x['Close']) / (x['High'] - x['Low']) - 1)

    return df


@app.route("/api/rest")
def index():
    return jsonify({
        'currencies_list_url': 'http://localhost:5000/api/rest/currencies',
        'currencies_get': 'http://localhost:5000/api/rest/currencies/<currency>'
    })


@app.route("/api/rest/currencies")
def currencies_list():
    return jsonify({})


@app.route("/api/rest/currencies/<currency>.<content_type>")
def currencies_get(currency, content_type):
    return create_response(get_historical_data(currency), content_type)


@app.route("/api/rest/currencies/<currency>/prediction")
def currencies_prediction_list(currency):
    return jsonify({'currency': currency})
