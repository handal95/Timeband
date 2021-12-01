import os
import numpy as np
import pandas as pd

from .utils.logger import Logger
from .utils.initiate import init_device
from torch.utils.data import DataLoader

from .loss import TIMEBANDLoss
from .model import TIMEBANDModel
from .metric import TIMEBANDMetric
from .dataset import TIMEBANDDataset
from .trainer import TIMEBANDTrainer
from .cleaner import TIMEBANDCleaner
from .predictor import TIMEBANDPredictor
from .dashboard import TIMEBANDDashboard


class TimebandBase:
    def __init__(self):
        self.device = init_device()
        self.logger = Logger("logs", 0)


class Timeband(TimebandBase):
    VERSION = "Timeband v0.0.0"

    def __init__(self, config):
        super().__init__()

        self.set_config(config)

        self.logger.info("**********************")
        self.logger.info("***** TIME  BAND *****")
        self.logger.info("**********************")

        self.dataset = TIMEBANDDataset(self.dataset_cfg)
        self.models = TIMEBANDModel(self.models_cfg)

        # Losses and Metric Settings
        self.metric = TIMEBANDMetric(self.metric_cfg)
        self.losses = TIMEBANDLoss(self.losses_cfg)

        # Visualize Settings
        self.dashboard = TIMEBANDDashboard(self.dashboard_cfg, self.dataset)

    def set_config(self, config: dict) -> None:
        """
        Setting configuration

        """
        # Set Logger
        config["core"]["device"] = self.device
        config["core"]["logger"] = self.logger
        config["core"]["targets_dims"] = len(config["dataset"]["targets"])

        # Configuration Categories
        self.__dict__ = {**config["core"], **self.__dict__}
        self.dataset_cfg = {**config["core"], **config["dataset"]}
        self.models_cfg = {**config["core"], **config["models"]}
        self.metric_cfg = {**config["core"], **config["dataset"]}
        self.losses_cfg = {**config["core"], **config["losses"]}
        self.trainer_cfg = {**config["core"], **config["trainer"]}
        self.dashboard_cfg = {**config["core"], **config["dashboard"]}
        self.cleaner_cfg = {**config["core"], **config["trainer"]}
        self.predictor_cfg = {**config["core"], **config["trainer"]}

        self.output_path = os.path.join(self.directory, self.data_name, self.TAG)
        self.cleanset_path = os.path.join(self.output_path, "cleaned_set.csv")
        self.bands_path = os.path.join(self.output_path, "bands_data.csv")
        self.predicted_path = os.path.join(self.output_path, "predicted_set.csv")

    def fit(self):
        self.models.initiate(dims=self.dataset.dims)

        self.trainer = TIMEBANDTrainer(
            self.trainer_cfg,
            self.dataset,
            self.models,
            self.metric,
            self.losses,
        )

        for k in range(self.dataset.sliding_step + 1):
            self.logger.info(f"Train ({k + 1}/{self.dataset.sliding_step + 1})")

            if self.pretrain:
                self.models.load("BEST")

            # Dataset
            trainset, validset = self.dataset.prepare_dataset(0, k + 1)
            trainset, validset = self.loader(trainset), self.loader(validset)

            # Model
            self.trainer.train(trainset, validset)

        pass

    def predict(self, data: pd.DataFrame) -> tuple((pd.DataFrame, pd.DataFrame)):
        # (Template) Predicting
        upper_index = [f"{idx}_upper" for idx in data.columns]
        lower_index = [f"{idx}_lower" for idx in data.columns]

        pred_lines = pd.DataFrame()
        pred_bands = pd.DataFrame()

        OBSERVED_LEN = 15
        FORECAST_LEN = 5
        for step in range(FORECAST_LEN):
            subdata = data[-OBSERVED_LEN - FORECAST_LEN + step : -FORECAST_LEN + step]

            pred_line = subdata.mean(axis=0)

            # Band
            pred_upper = subdata.mean(axis=0) * 1.02
            pred_upper.index = upper_index

            pred_lower = subdata.mean(axis=0) * 0.97
            pred_lower.index = lower_index

            pred_band = pd.concat([pred_lower, pred_upper], names=step)

            pred_lines = pd.concat([pred_lines, pred_line], axis=1)
            pred_bands = pd.concat([pred_bands, pred_band], axis=1)

        return (pred_lines.T, pred_bands.T)

    def predicts(self):
        if self.pretrain:
            self.models.load("BEST")

        self.cleaner = TIMEBANDCleaner(
            self.cleaner_cfg,
            self.dataset,
            self.models,
            self.losses,
            self.metric,
        )

        # Dataset
        dataset = self.dataset.prepare_dataset(split=False)
        dataset = self.loader(dataset)

        cleaned_data, band_data = self.cleaner.predicts(dataset)
        cleaned_data.to_csv(self.cleanset_path)
        band_data.to_csv(self.bands_path)

        return cleaned_data, band_data

    def loader(self, dataset: TIMEBANDDataset, batch_size=None) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size

        dataloader = DataLoader(dataset, batch_size, num_workers=self.workers)
        return dataloader
