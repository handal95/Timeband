import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.color import colorstr


def parsing(data: pd.DataFrame, config: dict, dt: pd.Series, name: str):
    if config[name] is True:
        time_info = pd.DataFrame({name: dt}, index=data.index)
        data = pd.concat([time_info, data], axis=1)

    return data


def fill_timegap(data: pd.DataFrame, time_index: str):
    print(data[time_index])
    TIMEGAP = (data[time_index].iloc[1] - data[time_index].iloc[0]).item()
        
    data_len = len(data)
    _tqdm = tqdm(range(1, data_len), desc="Check Timegap")

    for i in _tqdm:
        cur_time = data[time_index].iloc[i].item()
        next_time = data[time_index].iloc[i - 1].item()
        
        if (cur_time - next_time) != TIMEGAP:
            gap_start = next_time
            gap_end = cur_time - TIMEGAP

            for i in range((gap_end - gap_start) // TIMEGAP):
                time = gap_start + (i + 1) * TIMEGAP
                data = data.append({time_index: time}, ignore_index=True)

    data = data.set_index(time_index).sort_index().reset_index()
    filled = len(data) - data_len

    if filled > 0:
        print(f"Filling Time Gap : {colorstr(filled)} : timegap : {TIMEGAP}")
    data.replace(pd.NaT, np.nan, inplace=True)
    return data
