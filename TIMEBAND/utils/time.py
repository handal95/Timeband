import numpy as np
import pandas as pd


def parsing(data: pd.DataFrame, config: dict, dt: pd.Series, name: str):
    if config[name] is True:
        time_info = pd.DataFrame({name: dt}, index=data.index)
        data = pd.concat([time_info, data], axis=1)

    return data


def fill_timegap(data: pd.DataFrame, time_index: str):
    TIMEGAP = data[time_index][1] - data[time_index][0]

    data_len = len(data)
    for i in range(1, data_len):
        if data[time_index][i] - data[time_index][i - 1] != TIMEGAP:
            gap_start = data[time_index][i - 1]
            gap_end = data[time_index][i] - TIMEGAP

            for i in range((gap_end - gap_start) // TIMEGAP):
                time = gap_start + (i + 1) * TIMEGAP
                data = data.append({time_index: time}, ignore_index=True)

    data = data.set_index(time_index).sort_index().reset_index()
    filled = len(data) - data_len

    print(f"Filling Time Gap :{filled} : timegap : {TIMEGAP}")
    data.replace(pd.NaT, np.nan, inplace=True)
    return data
