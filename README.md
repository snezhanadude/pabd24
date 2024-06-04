# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.  

## Installation 

Клонируйте репозиторий, создайте виртуальное окружение, активируйте и установите зависимости:  

```sh
git clone https://github.com/yourgit/pabd24
cd pabd24
python -m venv venv

source venv/bin/activate  # mac or linux
.\venv\Scripts\activate   # windows

pip install -r requirements.txt
```

## Usage

### 1. Сбор данных о ценах на недвижимость

Файл `parse_cian.py` собирает данные о продаже квартир на сайте cian.ru. Выводит результаты в CSV файлы в папке data/raw.  
Параметры по умолчанию: сбор информации по 1 комнатным квартирам на первых двух страницах.
```bash
python src/parse_cian.py
```

### 2. Выгрузка данных в хранилище S3

Файл `upload_to_s3.py` загружает указанные файлы в хранилище S3. Требует наличия файла .env с ключами доступа AWS в корне проекта.  
Параметры:
- -i, --input - список локальных файлов данных для загрузки в S3 (по умолчанию указаны в скрипте).
```bash
python src/upload_to_s3.py -i data/raw/file1.csv data/raw/file2.csv
```

### 3. Загрузка данных из S3 на локальную машину

Файл `download_from_s3.py` скачивает файлы из S3 хранилища обратно на локальный диск.  
Параметры:
- -i, --input - список файлов в S3 для загрузки на локальную машину (по умолчанию указаны в скрипте).
```bash
python src/download_from_s3.py -i data/raw/file1.csv data/raw/file2.csv
```

### 4. Предварительная обработка данных

Файл `preprocess_data.py` преобразует сырые данные в датасеты для обучения и валидации. Записывает результаты в папку data/proc.  
Параметры:
- -s, --split - доля данных, идущая на обучающий набор (по умолчанию 0.9).
- -i, --input - список CSV файлов для обработки (по умолчанию указаны в скрипте).
```bash
python src/preprocess_data.py -s 0.9 -i data/raw/file1.csv data/raw/file2.csv
```

### 5. Обучение модели 

Файл `train_model.py` производит обучение модели и сохранение контрольной точки. Для предсказания цены недвижимости используется линейная регрессия. Модель обучается на данных о площади квартиры и её цене.  
Параметры:
- -m, --model - путь для сохранения обученной модели (по умолчанию 'models/linear_regression_v01.joblib').
```bash
python src/train_model.py -m models/linear_regression_v01.joblib
```

### 6. Запуск приложения flask 

Файл `predict_app.py` запускает веб-сервис предсказания цен на недвижимость. Веб-сервис на Flask предоставляет API для предсказания цен на основе данных о площади квартиры. Используется аутентификация через токены, которые хранятся в файле `.env`.
```bash
python src/predict_app.py
```
Адрес приложения: http://192.144.14.187:8000.  
Токен для авторизации: pabd24.

### 7. Использование сервиса через веб интерфейс

Для использования сервиса необходимо открыть файл `web/index.html` в браузере и ввести параметры квартиры и токен доступа. В форме нужно указать площадь квартиры (в квадратных метрах) и токен для доступа к сервису. После нажатия на кнопку "Submit" будет отправлен запрос на сервер, а результат (предсказанная цена квартиры) будет отображен на странице. Если запрос не будет выполнен успешно, выведется сообщение об ошибке.