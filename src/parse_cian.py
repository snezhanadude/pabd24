"""  Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""
import datetime

import cianparser
import pandas as pd

moscow_parser = cianparser.CianParser(location="Москва")


def main():
    """
    Function docstring
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = 1
    csv_path = f'data/raw/{n_rooms}_{t}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=True,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


if __name__ == '__main__':
    main()