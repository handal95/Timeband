import os
import pandas as pd

from torch.utils.data import DataLoader
from .loss import TIMEBANDLoss
from .model import TIMEBANDModel
from .metric import TIMEBANDMetric
from .dataset import TIMEBANDDataset
from .trainer import TIMEBANDTrainer
from .cleaner import TIMEBANDCleaner
from .predictor import TIMEBANDPredictor
from .dashboard import TIMEBANDDashboard

logger = None


class TIMEBANDCore:
    """
    TIMEBAND : Improving quality of Time series dataset based on LSTM-GAN
        - train()   : Pattern learning and Predicting of data
        - clean()   : Imputating Missing value and adjusting Anomaly value in data
                        for improving quality of raw data
        - predict() : Prediciting Future data

    """

    def __init__(self, config: dict) -> None:

        # Set Config
        self.set_config(config=config)

        # Dataset & Model Settings
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
        global logger
        logger = config["core"]["logger"]
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

    def train(self) -> None:
        # Init the models
        self.models.initiate(dims=self.dataset.dims)

        self.trainer = TIMEBANDTrainer(
            self.trainer_cfg,
            self.dataset,
            self.models,
            self.metric,
            self.losses,
            self.dashboard,
        )

        for k in range(self.dataset.sliding_step + 1):
            logger.info(f"Train ({k + 1}/{self.dataset.sliding_step + 1})")

            if self.pretrain:
                self.models.load("BEST")

            # Dataset
            trainset, validset = self.dataset.prepare_dataset(0, k + 1)
            trainset, validset = self.loader(trainset), self.loader(validset)

            # Model
            self.trainer.train(trainset, validset)

            logger.info(f"Done ({k + 1}/{self.dataset.sliding_step + 1}) ")

    def clean(self) -> None:
        # Init the models
        self.models.initiate(dims=self.dataset.dims)
        if self.pretrain:
            self.models.load("BEST")

        self.cleaner = TIMEBANDCleaner(
            self.cleaner_cfg,
            self.dataset,
            self.models,
            self.losses,
            self.metric,
            self.dashboard,
        )

        # Dataset
        dataset = self.dataset.prepare_dataset(split=False)
        dataset = self.loader(dataset)

        cleaned_data, band_data = self.cleaner.clean(dataset)
        cleaned_data.to_csv(self.cleanset_path)
        band_data.to_csv(self.bands_path)

        logger.info(f"Cleaned Data saved at {self.cleanset_path}")
        logger.info(f"Bands Data saved at {self.bands_path}")

    def predict(self):
        # Init the models
        self.models.initiate(dims=self.dataset.dims)
        if self.pretrain:
            self.models.load("BEST")

        self.predictor = TIMEBANDPredictor(
            self.cleaner_cfg,
            self.dataset,
            self.models,
            self.losses,
            self.metric,
            self.dashboard,
        )

        dataset = self.dataset.prepare_dataset(split=False)
        dataset = self.loader(dataset, batch_size=1)

        predicted_data = self.predictor.predict(dataset)
        predicted_data.to_csv(self.predicted_path)

        logger.info(f"Predicted Data saved at {self.predicted_path}")

    def loader(self, dataset: TIMEBANDDataset, batch_size=None) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size

        dataloader = DataLoader(dataset, batch_size, num_workers=self.workers)
        return dataloader
