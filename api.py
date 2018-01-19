import pandas as pd
import time
from flask import Flask, jsonify

app = Flask(__name__)


def get_historical_data(currency):
    start = "20130428"
    end = time.strftime("%Y%m%d")
    url = "https://coinmarketcap.com/currencies/" + currency + "/historical-data/?start=" + start + "&end=" + end
    items = pd.read_html(url)

    return items[0]


@app.route("/api/rest")
def index():
    return jsonify({
        'currencies_list_url': 'http://localhost:5000/api/rest/currencies',
        'currencies_get': 'http://localhost:5000/api/rest/currencies/<currency>'
    })


@app.route("/api/rest/currencies")
def currencies_list():
    return jsonify({})


@app.route("/api/rest/currencies/<currency>")
def currencies_get(currency):
    json = get_historical_data(currency).to_json(orient='records')

    return json
