import argparse
import logging
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

MODEL_SAVE_PATH = 'models/lightgbm_regression_v01.joblib'
TEST_DATA = 'data/proc/val.csv'

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/test_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

def main(args):
    df_test = pd.read_csv(TEST_DATA)
    X_test = df_test[['total_meters', 'floor', 'floors_count', 'district']]
    y_test = df_test['price']

    # Преобразование категориального признака в тип category
    X_test['district'] = X_test['district'].astype('category')

    model = load(args.model)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f'Test model {args.model} on {TEST_DATA}, MAE = {mae:.0f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
