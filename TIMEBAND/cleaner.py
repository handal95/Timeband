import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.color import colorstr
from TIMEBAND.loss import TIMEBANDLoss
from TIMEBAND.model import TIMEBANDModel
from TIMEBAND.metric import TIMEBANDMetric
from TIMEBAND.dataset import TIMEBANDDataset
from TIMEBAND.dashboard import TIMEBANDDashboard

logger = None

# Anomaly Labels
UPPER_ANOMALY = -1
MISSING_VALUE = 0
LOWER_ANOMALY = 1


class TIMEBANDCleaner:
    def __init__(
        self,
        config: dict,
        dataset: TIMEBANDDataset,
        models: TIMEBANDModel,
        metric: TIMEBANDMetric,
        losses: TIMEBANDLoss,
        dashboard: TIMEBANDDashboard,
    ) -> None:
        global logger
        logger = config["logger"]

        self.dataset = dataset
        self.models = models
        self.metric = metric
        self.losses = losses
        self.dashboard = dashboard

        # Set Config
        config = self.set_config(config=config)
        self.forecast_len = self.dataset.forecast_len

    def set_config(self, config: dict = None) -> dict:
        """
        Configure settings related to the data set.

        params:
            config: Trainer configuration dict
                `config['trainer']`
        """

        # Train option
        self.__dict__ = {**config, **self.__dict__}

    def clean(self, dataset: DataLoader) -> None:
        logger.info("RUN the model")

        # Prediction
        self.idx = 0
        # self.data_labeling()
        self.pred_initate()

        # Dashboard
        self.dashboard.init_figure()

        # Process step
        def generate(x):
            return self.models.netG(x)[:, : self.forecast_len].to(self.device)

        tqdm_ = tqdm(dataset)
        outputs = self.dataset.observed
        lower_bands = self.dataset.observed
        upper_bands = self.dataset.observed

        for i, data in enumerate(tqdm_):
            true_x = data["encoded"].to(self.device)
            true_y = data["decoded"].to(self.device)
            (batchs, forecast_len, target_dims) = true_y.shape

            # #######################
            # Generate
            # #######################
            fake_y = generate(true_x)

            # #######################
            # Process
            # #######################
            pred_y = self.dataset.denormalize(fake_y.cpu())
            preds, lower, upper = self.predicts(pred_y)

            pred_len = preds.shape[0]
            reals = self.dataset.forecast[self.idx : self.idx + pred_len].numpy()
            masks = self.dataset.missing[self.idx : self.idx + pred_len]

            output = np.concatenate([outputs[-1:], reals])
            target = self.adjust(output, preds, masks, lower, upper)
            lower_bands = np.concatenate([lower_bands[:1 - forecast_len], lower])
            upper_bands = np.concatenate([upper_bands[:1 - forecast_len], upper])
            outputs = np.concatenate([outputs[: 1 - forecast_len], target])

            # #######################
            # Visualize
            # #######################
            if i > 8:
                self.dashboard.vis(batchs, reals, preds, lower, upper, target)
            self.idx += batchs

        # Dashboard
        self.dashboard.clear_figure()

        # OUTPUTS
        lower_cols = [f"{x}_lower" for x in self.dataset.targets]
        upper_cols = [f"{x}_upper" for x in self.dataset.targets]

        index = self.dataset.times
        outputs_df = pd.DataFrame(outputs, columns=self.dataset.targets, index=index)

        bands_df = pd.concat(
            [pd.DataFrame(lower_bands, columns=lower_cols, index=index),
            pd.DataFrame(upper_bands, columns=upper_cols, index=index)],
            axis=1
        )
        bands_df.index.name = self.dataset.time_index
        outputs_df.index.name = self.dataset.time_index

        return outputs_df, bands_df

    def adjust(self, output, preds, masks, lower, upper):
        len = preds.shape[0]
        a = self.missing_gamma
        b = self.anomaly_gamma

        for p in range(len):
            value = output[p + 1]

            lmask = value < lower[p]
            umask = value > upper[p]
            mmask = masks[p]

            value = (1 - lmask) * value + lmask * (b * preds[p] + (1 - b) * value)
            value = (1 - umask) * value + umask * (b * preds[p] + (1 - b) * value)
            value = (1 - mmask) * value + mmask * (a * preds[p] + (1 - a) * output[p])

            output[p + 1] = value

        target = output[1:]
        return target

    def pred_initate(self):
        forecast_len = self.dataset.decode_shape[1]
        target_dims = self.dataset.decode_shape[2]

        self.preds = np.empty((forecast_len - 1, forecast_len, target_dims))
        self.preds[:] = np.nan

    def predicts(self, pred):
        (batch_size, forecast_len, target_dims) = pred.shape
        pred = pred.detach().numpy()

        nan_shape = np.empty((batch_size, forecast_len, target_dims))
        nan_shape[:] = np.nan

        self.preds = np.concatenate([self.preds[1 - forecast_len :], nan_shape])
        for f in range(forecast_len):
            self.preds[f : batch_size + f, f] = pred[:, f]

        preds = np.nanmedian(self.preds, axis=1)
        std = np.nanstd(self.preds, axis=1)

        for f in range(forecast_len - 1, 0, -1):
            gamma = (forecast_len - f) / (forecast_len - 1)
            std[-f] += std[-f - 1] * gamma

        lower = preds - self.band_width * std
        upper = preds + self.band_width * std

        return preds, lower, upper

    def data_labeling(self):
        if not self.labeling:
            return

        self.label_data = np.empty(self.target_data.shape)
        self.label_data[:] = np.nan
        self.outputs = self.target_data.to_numpy()
        self.labels = pd.DataFrame(
            self.label_data,
            columns=self.target_col,
            index=self.dataset.times,
        )

        if self.zero_is_missing:
            self.labels[self.target_data == 0] = MISSING_VALUE
            logger.info(f"A value of 0 is recognized as a missing value.")

        labels_path = os.path.join(self.directory, f"{self.data_name}_label.csv")
        self.labels.to_csv(labels_path)

        logger.info(f"CSV saved at {labels_path}")
