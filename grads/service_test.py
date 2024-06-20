"""
Test prediction web-service
If you have error: 503 Response, disable proxies in your requests
https://stackoverflow.com/questions/40430799/503-reponse-when-trying-to-use-python-request-on-local-website
"""
import logging
import time
import numpy as np
import requests
import tqdm
from dotenv import dotenv_values
import pandas as pd
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/service_test.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(message)s')

endpoints = [
    # endpoint = 'http://192.144.13.190:8000/predict' #
    # endpoint = 'http://192.144.12.8:8000/predict' #  Шувариков
    # endpoint = 'http://192.144.12.11:8000/predict' # Ахметов
    # ('http://192.144.14.11:5000/predict', 'Куличенко'),
    # ('http://192.144.14.9:8000/predict', 'Ланцова'),
    # ('http://192.144.12.199:8000/predict', 'Зверев'),
    # ('http://176.123.163.78:8000/predict', 'Пойкалайнен'),
    ('http://192.144.14.187:8000/predict', 'Дудкина'),
    # endpoint = 'http://192.144.14.8:8000/predict'  # Самсонкин
]

config = dotenv_values(".env")
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


def do_request(data: dict, endpoint) -> tuple:
    t0 = time.time()
    resp = requests.post(
        endpoint,
        json=data,
        headers=HEADERS
    )
    print(resp.text)
    resp = resp.json()
    t = time.time() - t0
    return t, resp['price']


def test_100(endpoint, name):
    df = pd.read_csv('data/test/test_100.csv')
    prices = df['price']
    df = df.drop(['price'], axis=1)
    records = df.to_dict('records')
    delays, pred_prices = [], []
    for row in tqdm.tqdm(records):
        t, price = do_request(row, endpoint)
        delays.append(t)
        # print(f'Price: {price}, delay: {t}')
        pred_prices.append(price)
    avg_delay = sum(delays) / len(delays)
    error = np.array(pred_prices) - prices.to_numpy()
    avg_error = np.mean(error)
    logger.info(f'{name}: Avg delay: {avg_delay*1000} ms, avg error: {avg_error} RUB')


if __name__ == '__main__':
    for endpoint, name in endpoints:
        test_100(endpoint, name)