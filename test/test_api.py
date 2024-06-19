import unittest
import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
ENDPOINT = 'http://192.144.14.187:8000'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}

class TestApi(unittest.TestCase):
    def test_home(self):
        resp = requests.get(ENDPOINT)
        self.assertIn('Housing price service', resp.text)

    def test_api(self):
        data = {'area': 42}
        resp = requests.post(ENDPOINT +'/predict',
                             json=data,
                             headers=HEADERS)
        self.assertIn('price', resp.text)


if __name__ == '__main__':
    unittest.main()