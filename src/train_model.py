import argparse
import logging
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
MODEL_SAVE_PATH = 'models/ridge_regression_v03.joblib'


def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    X = df_train[['total_meters', 'floor', 'floors_count', 'district']]
    y = df_train['price']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Определение колонок для One-Hot Encoding и стандартизации
    categorical_features = ['district']
    numeric_features = ['total_meters', 'floor', 'floors_count']

    # Создание трансформера для предобработки данных
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])

    # Создание и обучение модели Ridge регрессии
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

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
