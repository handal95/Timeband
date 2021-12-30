import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from typing import List, Tuple, Union, Optional

from utils.initiate import check_dirs_exist
from utils.initiate import init_device
from utils.time import fill_timegap
from utils.color import colorstr

class MyDataset(data.Dataset):
    def __init__(self, encoded, decoded):
        super(MyDataset, self).__init__()
        self.encoded = encoded
        self.decoded = decoded

        self.length = len(self.encoded)
        
    def shape(self, target="encode"):
        return self.encoded.shape if target == "encode" else self.decoded.shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        encoded = torch.tensor(self.encoded[idx], dtype=torch.float32)
        decoded = torch.tensor(self.decoded[idx], dtype=torch.float32)

        return encoded, decoded


class TimeData:
    def __init__(
        self,
        basedir: str,
        filename: str,
        targets: List[str],
        drops: List[str],
        fill_timegap: bool,
        time_index: str,
        time_encode: List[str],
        split_size: Union[int, float],
        observed_len: int,
        forecast_len: int,
    ) -> None:
        super(TimeData, self).__init__()
        self.device = init_device()

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
        self.datadir = os.path.join(self.basedir, "target")
        self.metadir = os.path.join(self.datadir, filename)
        check_dirs_exist(basepath=self.basedir, dirlist=["target"])
        check_dirs_exist(basepath=self.datadir, dirlist=[filename])

        self.base_path = os.path.join(self.basedir, f"{filename}.csv")
        self.data_path = os.path.join(self.metadir, f"{filename}.csv")
        self.minmax_path = os.path.join(self.metadir, "minmax.csv")
        self.missing_path = os.path.join(self.metadir, "missing.csv")

        # Dimension
        data = pd.read_csv(
            self.base_path,
            parse_dates=self.time_index,
            index_col=self.time_index,
            nrows=1,
        )
        data = self.parse_timeinfo(data)
        data.drop(self.drops, axis=1, inplace=True)
        
        self.encode_dims = len(data.columns)
        self.decode_dims = len(self.targets)

    def init_dataset(
        self,
        index_s: Optional[int] = 0,
        index_e: Optional[int] = None,
    ) -> pd.DataFrame:
        if os.path.exists(self.data_path):
            data = pd.read_csv(self.data_path, parse_dates=self.time_index)
        else:
            data = pd.read_csv(self.base_path, parse_dates=self.time_index)
            data.drop(self.drops, axis=1, inplace=True)
            data = data[index_s:index_e]
            
            data = fill_timegap(data, self.time_index) if self.fill_timegap else data
            data.to_csv(self.data_path, index=False)

        _data = data.set_index(self.time_index)
        _data = self.parse_timeinfo(_data)
        data = data.interpolate(method="ffill")
        data = data.interpolate(method="bfill")
        data.to_csv(self.data_path, index=False)

        # Dimension
        self.observed = _data
        self.forecast = _data[self.targets]

        # Missing Label
        if not os.path.exists(self.missing_path):
            missing_label = _data.isna().astype(int)
            missing_label.to_csv(self.missing_path)

        # Minmax information
        if not os.path.exists(self.minmax_path):
            minmax_label = self.minmax_info(_data)
            minmax_label.to_csv(self.minmax_path, index=False)

        return data

    def prepare_trainset(
        self, data: pd.DataFrame, stride: int = 1
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        # Dataset Preparing
        try:
            data.set_index(self.time_index, inplace=True)
        except:
            pass

        data = data.interpolate(method="ffill")
        data = data.interpolate(method="bfill")
        data.replace(np.nan, 0, inplace=True)
        
        data = self.parse_timeinfo(data)
        data = self.normalize(data)

        with open(self.missing_path) as f:
            missing_label = pd.read_csv(self.missing_path, index_col=self.time_index)

        self.missing_encode = missing_label
        self.missing_decode = missing_label[self.targets]

        # Data Windowing
        x, y = data, data[self.targets]

        observed, forecast = [], []
        stop = len(data) - self.forecast_len + 1
        for i in range(self.observed_len, stop, stride):
            observed.append(x[i - self.observed_len : i])
            forecast.append(y[i : i + self.forecast_len])

        encoded, decoded = np.array(observed), np.array(forecast)
        self.dimensions = x.shape[-1], y.shape[-1]

        split_index = len(data) - self.split_size

        if type(self.split_size) is float:
            split_index = int(len(data) * self.split_size)

        split_index = min(
            split_index, len(data) - self.observed_len - self.forecast_len
        )
        
        train_size = split_index
        valid_size = len(data) - split_index
        print(f"Split index is {train_size} / {valid_size} ({self.split_size}) ")

        trainset = MyDataset(encoded[:split_index], decoded[:split_index])
        validset = MyDataset(encoded[split_index:], decoded[split_index:])
        return trainset, validset

    def prepare_predset(self, data: pd.DataFrame, stride: int = 1):
        try:
            data.set_index(self.time_index, inplace=True)
        except:
            pass
        data = self.parse_timeinfo(data)
        data = self.normalize(data)

        missing_label = data.isna().astype(int)
        self.missing_encode = missing_label
        self.missing_decode = missing_label[self.targets]

        x, y = data, data[self.targets]

        observed, forecast = [], []
        stop = len(data) - self.forecast_len + 1
        for i in range(self.observed_len, stop, stride):
            observed.append(x[i - self.observed_len : i])
            forecast.append(y[i : i + self.forecast_len])

        encoded, decoded = np.array(observed), np.array(forecast)
        
        predset = MyDataset(encoded, decoded)
        return predset

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

        self.decode_min = torch.tensor(data_min[self.targets]).to(self.device)
        self.decode_max = torch.tensor(data_max[self.targets]).to(self.device)
        self.delta = torch.tensor(data_max[self.targets] - data_min[self.targets]).to(
            self.device
        )

        return data

    def denormalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Revert [-1,1] normalization"""
        if not hasattr(self, "decode_max") or not hasattr(self, "decode_min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        data = 0.5 * (data + 1)
        data = data * self.delta
        data = data + self.decode_min

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

    def get_random(self, dataset: MyDataset) -> tuple((torch.tensor, torch.tensor)):
        """
        Get Random data in trainset for `critic`

        """
        rand_scope = len(dataset) - self.forecast_len
        idx = np.random.randint(rand_scope)

        data = dataset[idx : idx + self.forecast_len]

        return data
