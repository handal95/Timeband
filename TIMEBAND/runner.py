import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.logger import Logger
from utils.color import colorstr
from TIMEBAND.loss import TIMEBANDLoss
from TIMEBAND.model import TIMEBANDModel
from TIMEBAND.metric import TIMEBANDMetric
from TIMEBAND.dataset import TIMEBANDDataset
from TIMEBAND.dashboard import TIMEBANDDashboard

logger = None

UPPER_ANOMALY = -1
MISSING_VALUE = 0
LOWER_ANOMALY = 1


class TIMEBANDRunner:
    def __init__(
        self,
        config: dict,
        dataset: TIMEBANDDataset,
        models: TIMEBANDModel,
        metric: TIMEBANDMetric,
        losses: TIMEBANDLoss,
        dashboard: TIMEBANDDashboard,
        device: torch.device,
    ) -> None:
        global logger
        logger = config["logger"]
        # Set device
        self.device = device

        self.dataset = dataset
        self.models = models
        self.metric = metric
        self.losses = losses
        self.dashboard = dashboard

        # Set Config
        config = self.set_config(config=config)

        self.data_name = self.dataset.data_name
        self.target_col = self.dataset.target_col
        self.target_data = self.dataset.data[self.target_col]
        self.observed_len = self.dataset.observed_len
        self.forecast_len = self.dataset.forecast_len

        self.labels = None

    def set_config(self, config: dict = None) -> dict:
        """
        Configure settings related to the data set.

        params:
            config: Trainer configuration dict
                `config['trainer']`
        """

        # Train option
        self.directory = config["directory"]
        self.labeling = config["labeling"]
        self.zero_is_missing = config["zero_is_missing"]

    def run(self, dataset: DataLoader) -> None:
        logger.info("RUN the model")

        # Label Setting
        self.data_labeling()

        # Prediction
        self.data_idx = 0
        self.pred_initate()

        # Dashboard
        self.dashboard.visual = True
        self.dashboard.init_figure()

        # Train Section
        for i, data in enumerate(tqdm(dataset)):
            true_x = data["encoded"].to(self.device)
            true_y = data["decoded"].to(self.device)
            (batchs, forecast_len, target_dims) = true_y.shape

            fake_y = self.models.netG(true_x)[:, :forecast_len].to(self.device)
            pred_y = self.dataset.denormalize(fake_y.cpu())

            self.predicts(pred_y)
            preds = torch.tensor(self.preds)

            reals = self.dataset.forecast[
                self.data_idx : self.data_idx + preds.shape[0]
            ].numpy()

            # Impute Zeros
            output = reals.copy()
            for b in range(batchs):
                label = self.label_data[self.observed_len + self.data_idx + b]
                rlabel = self.label_data[self.observed_len + self.data_idx + b - 1]
                output[b, label == MISSING_VALUE] = (
                    0.2 * preds[b, label == MISSING_VALUE]
                    + 0.8 * output[b - 1, label == MISSING_VALUE]
                )

            self.data_idx += batchs
            self.dashboard.train_vis(batchs, reals, self.preds, self.stds, output)

        # Dashboard
        self.dashboard.clear_figure()

        self.models.load("BEST")
        self.models.save(best=True)
        return None

    def pred_initate(self):
        decoded_shape = self.dataset.decode_shape
        (batch_size, forecast_len, target_dims) = decoded_shape

        init_shape3 = (forecast_len - 1, forecast_len, target_dims)
        init_shape2 = (forecast_len - 1, target_dims)

        # version v2.2
        self.pred_data = np.empty(init_shape3)
        self.pred_data[:] = np.nan

        self.preds = np.empty(init_shape2)
        self.preds[:] = np.nan

        self.stds = np.zeros(init_shape2)

    def predicts(self, pred):
        (batch_size, forecast_len, target_dims) = pred.shape
        pred = pred.detach().numpy()

        nan_shape3 = np.empty((batch_size, forecast_len, target_dims))
        nan_shape3[:] = np.nan
        nan_shape2 = np.empty((batch_size, target_dims))
        nan_shape2[:] = np.nan

        self.pred_data = np.concatenate(
            [self.pred_data[1 - forecast_len :], nan_shape3]
        )
        for f in range(forecast_len):
            self.pred_data[f : batch_size + f, f] = pred[:, f]

        self.preds = np.nanmedian(self.pred_data, axis=1)
        self.stds = np.nanstd(self.pred_data, axis=1)

        for f in range(forecast_len - 1, 0, -1):
            gamma = (forecast_len - f) / (forecast_len - 1)
            self.stds[-f] += self.stds[-f - 1] * gamma

        for f in range(1, forecast_len):
            self.stds[f] += self.stds[f - 1] * 0.1

    def data_labeling(self):
        if not self.labeling:
            return

        self.label_data = np.empty(self.target_data.shape)
        self.label_data[:] = np.nan
        self.outputs = self.target_data.to_numpy()
        self.labels = pd.DataFrame(
            self.label_data,
            columns=self.target_col,
            index=self.dataset.data.index,
        )

        if self.zero_is_missing:
            self.labels[self.target_data == 0] = MISSING_VALUE
            logger.info(f"A value of 0 is recognized as a missing value.")

        labels_path = os.path.join(self.directory, f"{self.data_name}_label.csv")
        self.labels.to_csv(labels_path)

        logger.info(f"CSV saved at {labels_path}")


def desc(training, epoch, score, losses):
    process = "Train" if training else "Valid"

    if not training:
        score["SCORE"] = colorstr("bright_red", score["SCORE"])
        score["RMSE"] = colorstr("bright_blue", score["RMSE"])
        score["NMAE"] = colorstr("bright_red", score["NMAE"])
        losses["L1"] = colorstr("bright_blue", losses["L1"])
        losses["L2"] = colorstr("bright_blue", losses["L2"])
        losses["GP"] = colorstr("bright_blue", losses["GP"])

    return (
        f"[{process} e{epoch + 1:4d}] "
        f"Score {score['SCORE']} ( NME {score['NME']} / NMAE {score['NMAE']} / RMSE {score['RMSE']} ) "
        f"D {losses['D']} ( R {losses['R']} F {losses['F']} ) "
        f"G {losses['G']} ( G {losses['G_']} L1 {losses['L1']} L2 {losses['L2']} GP {losses['GP']} )"
    )
