import torch
from utils.logger import Logger
from utils.device import init_device

from torch.utils.data import DataLoader
from TIMEBAND.loss import TIMEBANDLoss
from TIMEBAND.model import TIMEBANDModel
from TIMEBAND.metric import TIMEBANDMetric
from TIMEBAND.dataset import TIMEBANDDataset
from TIMEBAND.trainer import TIMEBANDTrainer
from TIMEBAND.runner import TIMEBANDRunner
from TIMEBAND.dashboard import TIMEBANDDashboard

logger = None


class TIMEBANDCore:
    """
    TIMEBANDBand : Timeseries Analysis using GAN Band

    The Model for Detecting anomalies / Imputating missing value in timeseries data

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
        self.runner_cfg = {**config["core"], **config["runner"]}

    def init_device(self):
        """
        Setting device CUDNN option

        """
        # TODO : Using parallel GPUs options
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def train(self) -> None:
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
            logger.info(f"Train ({k + 1}/{self.dataset.sliding_step})")

            if self.pretrain:
                self.models.load("BEST")

            # Dataset
            trainset, validset = self.dataset.prepare_dataset(k + 1)
            trainset, validset = self.loader(trainset), self.loader(validset)

            # Model
            self.trainer.train(trainset, validset)

            logger.info(f"Done ({k + 1}/{self.dataset.sliding_step + 1}) ")

    def run(self):
        self.models.initiate(dims=self.dataset.dims)
        if self.pretrain:
            self.models.load("BEST")
        self.runner = TIMEBANDRunner(
            self.runner_cfg,
            self.dataset,
            self.models,
            self.losses,
            self.metric,
            self.dashboard,
            self.device,
        )

        self.batch_size = 1
        dataset = self.dataset.prepare_testset(0, split=False)
        dataset = self.loader(dataset)

        output = self.runner.run(dataset)
        output.to_csv(f"./outputs/output.csv", index=False)

    def visualize(self):
        pass

    def loader(self, dataset: TIMEBANDDataset):
        dataloader = DataLoader(dataset, self.batch_size, num_workers=self.workers)
        return dataloader

    def clear(self):
        del self.dataset
        self.dataset = None
