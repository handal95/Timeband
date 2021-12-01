import os
from pandas.core.frame import DataFrame
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
from .utils.dataset import Dataset
from .utils.time import parsing, fill_timegap

logger = None


class TIMEBANDDataset:
    """
    TIMEBAND Dataset

    """

    def __init__(self, config: dict) -> None:
        """
        TIMEBAND Dataset

        params:
            config: Dataset configuration dict
        """

        # Set Config
        self.set_config(config)

        # Init Dataset
        self.init_dataset()

        # Load Data
        self.data = self.load_dataset()

        # Information
        self.logger.info(
            f"\n  Dataset: \n"
            f"  - Config    : {config} \n"
            f"  - Time Idx  : {self.time_index} \n"
            f"  - Length    : {self.data_length} \n"
            f"  - Shape(E/D): {self.encode_shape} / {self.decode_shape} \n"
            f"  - Targets   : {self.targets} ({self.decode_dim} cols) \n",
            level=0,
        )

    def set_config(self, config: dict) -> None:
        """
        Configure settings related to the data set.

        params:
            config: Dataset configuration dict
                `config['core'] & config['dataset']`
        """

        # Data file configuration
        self.__dict__ = {**config, **self.__dict__}
        self.basepath = os.path.join(self.directory, self.data_name)

        self.datapath = os.path.join(self.basepath, "original_data.csv")
        self.missing_path = os.path.join(self.basepath, "missing_label.csv")
        self.anomaly_path = os.path.join(self.basepath, "anomaly_label.csv")
        self.minmax_path = os.path.join(self.basepath, "minmax_info.csv")
        self.normalized_path = os.path.join(self.basepath, "normalized.csv")

    def init_dataset(self) -> None:
        """
        Prepare data and labels for train/analysis

        """

        ##################
        # Data Preparing #
        ##################
        if not os.path.exists(self.datapath):
            # Read csv data
            csv_path = os.path.join(self.directory, f"{self.data_name}.csv")
            data = pd.read_csv(csv_path, parse_dates=[self.time_index])
            data.drop(self.drops, axis=1, inplace=True)

            # Time indexing
            data = fill_timegap(data, self.time_index) if self.fill_timegap else data
            data = self.expand_data(data)

            data.set_index(self.time_index, inplace=True)
            data.sort_index(ascending=True, inplace=True)

            data = self.parse_timeinfo(data)

            self.minmax_info(data)
            data.to_csv(self.datapath)

            self.logger.info(f"  Data has been saved to {self.datapath}")

        if not os.path.exists(self.missing_path):
            # Missing values Labeling
            missing_label = data.isna().astype(int)
            missing_label.to_csv(self.missing_path)
            self.logger.info(f"  Missing Label has been saved to {self.missing_path}")

        if not os.path.exists(self.anomaly_path):
            # Anomalies Labeling
            anomaly_label = pd.DataFrame(
                np.zeros(data.shape), columns=data.columns, index=data.index
            )
            anomaly_label.to_csv(self.anomaly_path)
            self.logger.info(f"  Anomaly Label has been saved to {self.anomaly_path}")

    def interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        data.interpolate(method="ffill", inplace=True)
        data.interpolate(method="bfill", inplace=True)
        data.replace(np.nan, 0, inplace=True)
        return data

    def load_dataset(self, datapath: str = None) -> pd.DataFrame:
        """
        Load timeseries dataset from {DATA_DIR}/{DATA_NAME}/{DATA_NAME}.csv
            if not file is exists, save the basic file

        """
        # Load Dataset
        datapath = self.datapath if datapath is None else datapath
        data = pd.read_csv(datapath, parse_dates=[self.time_index])
        data.set_index(self.time_index, inplace=True)
        data = self.interpolate(data)

        # Observed Data & Forecast Data
        observed = data[self.targets][: self.observed_len + self.forecast_len - 1]
        forecast = data[self.targets][self.observed_len :]
        self.observed = torch.from_numpy(observed.to_numpy())
        self.forecast = torch.from_numpy(forecast.to_numpy())

        # Label information
        missing_label = pd.read_csv(self.missing_path)[self.targets]
        anomaly_label = pd.read_csv(self.anomaly_path)[self.targets]
        self.missing = missing_label[self.observed_len :].to_numpy()
        self.anomaly = anomaly_label[self.observed_len :].to_numpy()

        # Data Processing
        # Min-Max Scaling : 2 * (x - x.min) / (x.max - x.min) - 1
        minmax = self.minmax_info(data)
        data = self.normalize(data, minmax)
        data.to_csv(self.normalized_path)

        # Data information
        self.times = data.index.strftime(self.time_format).tolist()
        self.data_length = data.shape[0]

        self.encode_dim = len(data.columns)
        self.decode_dim = len(self.targets)

        self.dims = {"encode": self.encode_dim, "decode": self.decode_dim}
        self.encode_shape = (self.batch_size, self.observed_len, self.encode_dim)
        self.decode_shape = (self.batch_size, self.forecast_len, self.decode_dim)

        return data

    def prepare_dataset(
        self, _from: int = 0, k_step: int = 0, split=True
    ) -> pd.DataFrame:
        # Prepare data
        sliding_len = self.data_length - self.sliding_step + k_step

        data = self.data[_from:sliding_len]
        data_len = len(data)

        # Windowing data
        stop = data_len - self.forecast_len
        encoded, decoded = self.windowing(data, stop)

        if not split:
            # Dataset Preparing
            dataset = Dataset(encoded, decoded)

            # Unsplited data shape info
            self.logger.info(f" - Data shape : {dataset.shape()}")
            return dataset

        # Splited dataset
        split_rate = self.split_rate
        valid_minlen = int((self.min_valid_scale) * self.forecast_len)
        valid_idx = min(int(data_len * split_rate), data_len - valid_minlen)
        split_idx = valid_idx - self.forecast_len - self.observed_len

        # Dataset Preparing
        trainset = Dataset(encoded[:split_idx], decoded[:split_idx])
        validset = Dataset(encoded[split_idx:], decoded[split_idx:])
        self.trainset = trainset

        # Feature info
        self.logger.info(f"- Split Rate : T {split_rate:.3f} V {1 - split_rate:.3f}")
        self.logger.info(
            f"- Train shape: T {trainset.shape()},  {trainset.shape('decode')}"
        )
        self.logger.info(
            f"- Valid shape: T {validset.shape()}, {validset.shape('decode')}"
        )

        return trainset, validset

    def windowing(self, x: pd.DataFrame, stop: int) -> tuple((np.array, np.array)):
        """
        Windowing data

        """

        y = x[self.targets]

        observed, forecast = [], []
        for i in range(self.observed_len, stop + 1, self.stride):
            observed.append(x[i - self.observed_len : i])
            forecast.append(y[i : i + self.forecast_len])

        observed, forecast = np.array(observed), np.array(forecast)

        return observed, forecast

    def normalize(self, data: pd.DataFrame, minmax: pd.DataFrame) -> pd.DataFrame:
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        # 2 * (x - x.min) / (x.max - x.min) - 1

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
            batch_denorm = data[batch]
            batch_denorm = 0.5 * (batch_denorm + 1)
            batch_denorm = batch_denorm * delta
            batch_denorm = batch_denorm + self.decode_min
            data[batch] = batch_denorm

        return data

    def get_random(self) -> tuple((torch.tensor, torch.tensor)):
        """
        Get Random data in trainset for `critic`

        """
        rand_scope = len(self.trainset) - self.forecast_len
        idx = np.random.randint(rand_scope)

        data = self.trainset[idx : idx + self.forecast_len]

        encoded = data["encoded"].to(self.device)
        decoded = data["decoded"].to(self.device)

        return encoded, decoded

    ###############
    # Init preprocessed methods
    ###############
    def expand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        (data_len, encode_dim) = data.shape
        expand_data = pd.DataFrame(
            np.zeros((self.forecast_len, encode_dim)),
            columns=data.columns,
        ).replace(0, np.nan)

        time_index = data[self.time_index]

        timegap = time_index[1] - time_index[0]
        time_index = data[self.time_index][data_len - 1]
        for i in range(self.forecast_len):
            time_index = time_index + timegap
            expand_data[self.time_index][i] = time_index

        data = pd.concat([data, expand_data], axis=0)

        return data

    def parse_timeinfo(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        From time index column, parse date-time properties

        """
        datetime = data.index  # datetime information

        data = parsing(data, self.time_encode, datetime.year, name="year")
        data = parsing(data, self.time_encode, datetime.month, name="month")
        data = parsing(data, self.time_encode, datetime.day, name="day")
        data = parsing(data, self.time_encode, datetime.weekday, name="weekday")
        data = parsing(data, self.time_encode, datetime.hour, name="hour")
        data = parsing(data, self.time_encode, datetime.minute, name="minute")

        return data

    def minmax_info(self, data: pd.DataFrame) -> None:
        """
        Set Min-Max information for data scaling

        """

        # the min-max value of the data to be actually received afterward is unknown
        # So, using min/max information only 90% of dataset
        # and give a small margin was set based on the observed values.

        split_idx = int(len(data) * self.split_rate)
        min_val = data[:split_idx].min()
        max_val = data[:split_idx].max()

        min_val[min_val == -np.inf] = max_val
        max_val[min_val == max_val] = max_val + 1

        minmax_df = pd.DataFrame([data.columns, min_val, max_val]).T

        self.logger.info(
            f"Min Max info\n{tabulate(minmax_df, headers='keys', floatfmt='.2f')}",
            level=0,
        )

        return {"min": min_val, "max": max_val}
