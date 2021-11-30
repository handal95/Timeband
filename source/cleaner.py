import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from .loss import TIMEBANDLoss
from .model import TIMEBANDModel
from .metric import TIMEBANDMetric
from .dataset import TIMEBANDDataset

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
    ) -> None:
        # Set Config
        self.set_config(config=config)

        self.dataset = dataset
        self.models = models
        self.metric = metric
        self.losses = losses

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

    def predicts(self, dataset: DataLoader) -> None:
        self.logger.info("RUN the model")

        # Prediction
        self.idx = 0
        self.pred_initate()

        # Process step
        def generate(x):
            return self.models.netG(x).to(self.device)

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
            lower_bands = np.concatenate([lower_bands[: 1 - forecast_len], lower])
            upper_bands = np.concatenate([upper_bands[: 1 - forecast_len], upper])
            outputs = np.concatenate([outputs[: 1 - forecast_len], target])

            # #######################
            # Visualize
            # #######################
            self.idx += batchs

        # OUTPUTS
        lower_cols = [f"{x}_lower" for x in self.dataset.targets]
        upper_cols = [f"{x}_upper" for x in self.dataset.targets]

        index = self.dataset.times
        outputs_df = pd.DataFrame(outputs, columns=self.dataset.targets, index=index)

        bands_df = pd.concat(
            [
                pd.DataFrame(lower_bands, columns=lower_cols, index=index),
                pd.DataFrame(upper_bands, columns=upper_cols, index=index),
            ],
            axis=1,
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

            mmask = masks[p]
            lmask = value < lower[p]
            umask = value > upper[p]

            value = (1 - mmask) * value + mmask * (a * preds[p] + (1 - a) * output[p])
            value = (1 - lmask) * value + lmask * (b * preds[p] + (1 - b) * value)
            value = (1 - umask) * value + umask * (b * preds[p] + (1 - b) * value)

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