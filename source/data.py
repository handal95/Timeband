import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import torch
import torch.utils.data as data
from typing import List, Tuple, Union, Optional
from .utils.time import fill_timegap


class Dataset(data.Dataset):
    def __init__(self, encoded, decoded):
        super(Dataset, self).__init__()
        self.encoded = encoded
        self.decoded = decoded

    def shape(self, target="encode"):
        return self.encoded.shape if target == "encode" else self.decoded.shape

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        data = {
            "encoded": torch.tensor(self.encoded[idx], dtype=torch.float32),
            "decoded": torch.tensor(self.decoded[idx], dtype=torch.float32),
        }

        return data


class TIMEBANDData:
    def __init__(
        self,
        basedir: str,
        filename: str,
        targets: List[str],
        drops: List[str],
        fill_timegap: bool,
        time_index: List[str],
        time_encode: List[str],
        split_size: Union[int, float],
        observed_len: int,
        forecast_len: int,
    ) -> None:
        # Basic Configuration
        self.basedir = basedir
        self.filename = filename
        self.targets = targets
        self.drops = drops
        self.fill_timegap = fill_timegap
        self.time_index = time_index
        self.time_encode = time_encode
        self.split_size = split_size
        self.observed_len = observed_len
        self.forecast_len = forecast_len

        # Path setting
        self.basepath = os.path.join(self.basedir, "origin")
        self.datapath = os.path.join(self.basedir, "target")
        self.metapath = os.path.join(self.basedir, "meta", filename)
        os.mkdir(self.datapath) if not os.path.exists(self.datapath) else None
        os.mkdir(self.metapath) if not os.path.exists(self.metapath) else None

        self.origin_path = os.path.join(self.basepath, f"{filename}.csv")
        self.target_path = os.path.join(self.datapath, f"{filename}.csv")
        self.minmax_path = os.path.join(self.metapath, "minmax.csv")
        self.missing_path = os.path.join(self.metapath, "missing.csv")

    def init_dataset(
        self,
        index_s: Optional[int] = 0,
        index_e: Optional[int] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        data = pd.read_csv(self.origin_path, parse_dates=self.time_index)
        data.drop(self.drops, axis=1, inplace=True)
        data = data[index_s:index_e]

        data = fill_timegap(data, self.time_index) if self.fill_timegap else data

        if force is False and os.path.exists(self.target_path):
            return data

        data.to_csv(self.target_path)
        _data = data.set_index(self.time_index)
        _data = self.parse_timeinfo(_data)

        # Missing Label
        missing_label = _data.isna().astype(int)
        missing_label.to_csv(self.missing_path)

        # Minmax information
        minmax_label = self.minmax_info(_data)
        minmax_label.to_csv(self.minmax_path, index=False)
        
        return data

    def prepare_dataset(
        self, data: Optional[pd.DataFrame] = None, stride: int = 1, split: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        # Dataset Preparing
        data.set_index(self.time_index, inplace= True)
        data = self.parse_timeinfo(data)
        data = self.normalize(data)

        # Data Windowing
        x, y = data, data[self.targets]

        observed, forecast = [], []
        stop = len(data) - self.forecast_len + 1
        for i in range(self.observed_len, stop, stride):
            observed.append(x[i - self.observed_len : i])
            forecast.append(y[i : i + self.forecast_len])

        encoded, decoded = np.array(observed), np.array(forecast)
        
        if split:
            split_index = len(data) - self.split_size

            if type(self.split_size) is float:
                split_index = int(len(data) * self.split_size)
                
            trainset = Dataset(encoded[:split_index], decoded[:split_index])
            validset = Dataset(encoded[split_index:], decoded[split_index:])
            return trainset, validset

        else:
            dataset = Dataset(encoded, decoded)
            return dataset

    def minmax_info(self, data: pd.DataFrame) -> None:
        """
        Set Min-Max information for data scaling

        """
        # the min-max value of the data to be actually received afterward is unknown
        # So, using min/max information only 90% of dataset
        # and give a small margin was set based on the observed values.
        split_index = int(len(data) * self.split_size)

        min_val = data[:split_index].min()
        max_val = data[:split_index].max()

        minmax_df = pd.DataFrame(
            [data.columns, min_val, max_val], ["Feats", "min", "max"]
        ).T
        return minmax_df

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        # 2 * (x - x.min) / (x.max - x.min) - 1
        
        minmax = pd.read_csv(self.minmax_path, index_col=["Feats"])

        data_min, data_max = minmax["min"], minmax["max"]

        data = 2 * ((data - data_min) / (data_max - data_min)) - 1

        self.decode_min = torch.tensor(data_min[self.targets])
        self.decode_max = torch.tensor(data_max[self.targets])

        return data

    def denormalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Revert [-1,1] normalization"""
        if not hasattr(self, "decode_max") or not hasattr(self, "decode_min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        delta = self.decode_max - self.decode_min
        for batch in range(data.shape[0]):
            data[batch] = 0.5 * (data[batch] + 1)
            data[batch] = data[batch] * delta
            data[batch] = data[batch] + self.decode_min

        return data

    def parse_timeinfo(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        From time index column, parse date-time properties

        """
        timeinfo = {
            "year": data.index.year,
            "month": data.index.month,
            "weekday": data.index.weekday,
            "day": data.index.day,
            "hour": data.index.hour,
            "minute": data.index.minute,
        }

        for target in self.time_encode:
            time_info = pd.DataFrame({target: timeinfo[target]}, index=data.index)
            data = pd.concat([time_info, data], axis=1)

        return data
