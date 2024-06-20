import argparse
import logging
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
MODEL_SAVE_PATH = 'models/lightgbm_regression_v01.joblib'


def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    X = df_train[['total_meters', 'floor', 'floors_count', 'district']]
    y = df_train['price']

    # Преобразование категориальных признаков в тип category
    X['district'] = X['district'].astype('category')

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train, categorical_feature=['district'])

    # Сохранение модели
    dump(model, args.model)
    logger.info(f'Saved to {args.model}')

    # Оценка модели на обучающей выборке
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    logger.info(f'Mean Absolute Error on train set: {mae_train:.3f}')

    # Оценка модели на тестовой выборке
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    logger.info(f'Mean Absolute Error on test set: {mae_test:.3f}')

    # Кросс-валидация
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_score = -cv_scores.mean()
    logger.info(f'Mean Cross-Validation MAE: {mean_cv_score:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
