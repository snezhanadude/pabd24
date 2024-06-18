import argparse
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
MODEL_SAVE_PATH = 'models/linear_regression_v01.joblib'


def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    X_train = df_train[['total_meters',
                        'floor',
                        'floors_count',
                        'district'
                        ]]
    y_train = df_train['price']

    # Определение колонок для One-Hot Encoding
    categorical_features = ['district']
    numeric_features = ['total_meters', 'floor', 'floors_count']

    # Создание трансформера для предобработки данных
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

    # Создание и обучение модели линейной регрессии
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    # Сохранение модели
    dump(model, args.model)
    logger.info(f'Saved to {args.model}')

    # Оценка модели
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)

    logger.info(f'Mean Absolute Error on train set: {mae:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
