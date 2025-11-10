import os
from datetime import datetime
import numpy as np


def get_timespatial_series(sea_name, start_date, stop_date):
    """
    Function for loading spatiotemporal data for sea
    """
    datamodule_path = '/path_to_data/'  # Пользователь должен заменить этот путь

    if not os.path.exists(datamodule_path):
        raise FileNotFoundError(
            f"Data path '{datamodule_path}' not found.\n"
            "Please make sure to:\n"
            "1. Download the data from: https://disk.yandex.ru/d/C8KrnPCr65nqSw\n"
            "2. Extract the .rar archive in 'kara\ '\n"
            "3. Update the 'datamodule_path' in this function with the correct path to EXTRACTED data (where .npy lies)"
        )
    files_path = datamodule_path

    timespatial_series = []
    dates_series = []
    for file in os.listdir(files_path):
        date = datetime.strptime(file, f'osi_%Y%m%d.npy')
        if start_date <= date.strftime('%Y%m%d') < stop_date:
            array = np.load(f'{files_path}/{file}')
            timespatial_series.append(array)
            dates_series.append(date)
        else:
            pass
    timespatial_series = np.array(timespatial_series)
    return timespatial_series, dates_series
