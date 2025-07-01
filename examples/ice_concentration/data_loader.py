import os
from datetime import datetime
import numpy as np


def get_timespatial_series(sea_name, start_date, stop_date):
    """
    Function for loading spatiotemporal data for sea
    """
    datamodule_path = '/path_to_data/'
    files_path = f'{datamodule_path}/{sea_name}'
    timespatial_series = []
    dates_series = []
    for file in os.listdir(files_path):
        date = datetime.strptime(file, f'osi_%Y%m%d.npy')
        if start_date <= date.strftime('%Y%m%d') < stop_date:
            array = np.load(f'{files_path}/{file}')
            timespatial_series.append(array)
            dates_series.append(date)
        else:
            break
    timespatial_series = np.array(timespatial_series)
    return timespatial_series, dates_series